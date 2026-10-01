"""NeuroScan CLI - Medical AI and Explainability Suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import typer
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from neuroscan.models import DiagnosticLabel
from neuroscan.predictor import NeuroPredictor
from neuroscan.preprocess import crop_brain_contour, load_image

app = typer.Typer(
    name="neuroscan",
    help="Deep Learning & Grad-CAM Explainable AI for Brain MRI Tumor Detection.",
    add_completion=False,
)
console = Console()


@app.command()
def predict(
    image_path: Path = typer.Argument(
        ...,
        help="Path to brain MRI scan image (.jpg, .jpeg, .png).",
    ),
    weights: Optional[Path] = typer.Option(
        None,
        "--weights",
        "-w",
        help="Path to trained VGG-16 weights file (.h5).",
    ),
    out_dir: Optional[Path] = typer.Option(
        Path("gradcam_outputs"),
        "--out-dir",
        "-o",
        help="Directory to save Grad-CAM and cropped visualization overlays.",
    ),
    save_viz: bool = typer.Option(
        True,
        "--save-viz/--no-save-viz",
        help="Save cropped, heatmap, and overlay images.",
    ),
) -> None:
    """Run diagnostic classification and Grad-CAM explainability on an MRI scan."""
    if not image_path.exists():
        console.print(f"[bold red]Error: Image not found:[/bold red] {image_path}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan][*] Analyzing MRI Scan:[/bold cyan] {image_path.name}")

    predictor = NeuroPredictor(weights_path=weights)
    report = predictor.predict(
        image_input=image_path,
        output_dir=out_dir,
        save_visualizations=save_viz,
    )

    is_tumor = report.label == DiagnosticLabel.TUMOR_DETECTED
    badge_style = "bold red" if is_tumor else "bold green"
    label_text = f"[{badge_style}]{report.label.value.upper()}[/{badge_style}]"

    table = Table(title="Diagnostic Assessment", show_header=True)
    table.add_column("Parameter", style="dim")
    table.add_column("Finding", style="bold")

    table.add_row("Diagnosis", label_text)
    table.add_row("Model Confidence", f"{report.confidence:.2%}")
    table.add_row("Tumor Probability", f"{report.raw_score:.2%}")
    if report.bbox:
        table.add_row("Cropped Parenchyma", f"{report.bbox.width}x{report.bbox.height} px")
    table.add_row("Clinical Impression", report.notes)

    console.print("\n", table)

    if save_viz and report.overlay_image_path:
        console.print(Panel(
            f"[green]✔ Visualizations successfully saved to:[/green] [cyan]{out_dir.resolve()}[/cyan]\n"
            f"  • Cropped Scan:  {report.cropped_image_path.name}\n"
            f"  • Grad-CAM Map:  {report.gradcam_image_path.name}\n"
            f"  • Diagnostic Overlay: [bold yellow]{report.overlay_image_path.name}[/bold yellow]",
            title="Explainable AI Telemetry",
            border_style="cyan",
        ))


@app.command()
def crop(
    image_path: Path = typer.Argument(
        ...,
        help="Path to MRI scan to crop.",
    ),
    out_path: Path = typer.Option(
        Path("cropped_preview.png"),
        "--out",
        "-o",
        help="Destination path for cropped image.",
    ),
) -> None:
    """Demonstrate extreme-point contour cropping (skull stripping)."""
    if not image_path.exists():
        console.print(f"[bold red]Error: Image not found:[/bold red] {image_path}")
        raise typer.Exit(code=1)

    img_rgb = load_image(image_path)
    cropped_rgb, bbox = crop_brain_contour(img_rgb)

    Image.fromarray(cropped_rgb).save(out_path)
    console.print(
        f"[green]✔ Cropped brain tissue saved to:[/green] [cyan]{out_path}[/cyan]\n"
        f"Original shape: {img_rgb.shape[1]}x{img_rgb.shape[0]} | "
        f"Cropped box: {bbox.width}x{bbox.height} at ({bbox.min_x}, {bbox.min_y})"
    )


@app.command()
def benchmark(
    dataset_dir: Path = typer.Option(
        Path("brain_tumor_dataset"),
        "--data-dir",
        "-d",
        help="Dataset path containing yes/ and no/ folders.",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Number of images per class to evaluate.",
    ),
) -> None:
    """Run diagnostic verification across sample dataset images."""
    if not dataset_dir.exists():
        console.print(f"[bold red]Error: Dataset directory not found:[/bold red] {dataset_dir}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold magenta][*] Running Benchmark on Sample Scans ({limit} per class)...[/bold magenta]\n")

    predictor = NeuroPredictor()
    classes = [("yes", DiagnosticLabel.TUMOR_DETECTED), ("no", DiagnosticLabel.HEALTHY)]

    total_tested = 0
    correct = 0

    table = Table(title="Sample Evaluation Results", show_header=True)
    table.add_column("Scan", style="dim")
    table.add_column("Ground Truth")
    table.add_column("Predicted")
    table.add_column("Confidence")
    table.add_column("Status")

    for folder_name, true_label in classes:
        folder_path = dataset_dir / folder_name
        if not folder_path.is_dir():
            continue

        images = [f for f in folder_path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")][:limit]
        for img_p in images:
            report = predictor.predict(img_p)
            is_correct = report.label == true_label
            if is_correct:
                correct += 1
            total_tested += 1

            status_str = "[green]MATCH[/green]" if is_correct else "[red]MISMATCH[/red]"
            table.add_row(
                img_p.name,
                true_label.value,
                report.label.value,
                f"{report.confidence:.1%}",
                status_str,
            )

    console.print(table)
    if total_tested > 0:
        acc = (correct / total_tested) * 100
        console.print(f"\n[bold]Sample Accuracy:[/bold] [green]{acc:.1f}%[/green] ({correct}/{total_tested} matches)\n")


if __name__ == "__main__":
    app()
