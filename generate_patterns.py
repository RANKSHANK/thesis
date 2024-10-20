import cairo
from pathlib import Path
from math import pi

def run():
    # Create pattern directory
    Path("./run/patterns").mkdir(exist_ok=True, parents=True)
    # Create Chessboard SVG
    cairo_sfc = cairo.SVGSurface(Path("./run/patterns/chessboard.svg").__str__(), 1024, 1024)
    cairo_ctx = cairo.Context(cairo_sfc)
    chessboard(cairo_ctx, cells_per_row=8)
    cairo_sfc.finish()
    cairo_sfc.flush()

    # Create Circleboard SVG
    cairo_sfc = cairo.SVGSurface(Path("./run/patterns/circleboard.svg").__str__(), 1024, 1024)
    cairo_ctx = cairo.Context(cairo_sfc)
    circleboard(cairo_ctx, cells_per_row=14)
    cairo_sfc.finish()
    cairo_sfc.flush()

def _draw_grid(cairo_ctx, pattern, count, pixels):
    ratio = pixels / count
    # create the base white background
    cairo_ctx.set_source_rgb(1.0, 1.0, 1.0)
    cairo_ctx.rectangle(0, 0, pixels, pixels)
    cairo_ctx.fill()
    # loop for alternate placing of black grid sections
    cairo_ctx.set_source_rgb(0, 0, 0)
    for row in range(count):
        for column in range(count):
            # alternating placement on (odd, odd) | (even, even) cells based on row
            if column & 1 == (row & 1):
                pattern(ctx=cairo_ctx, x=column * ratio, y=row * ratio)
                cairo_ctx.fill()

def chessboard(cairo_ctx, cells_per_row=10, pixels_per_row=1024):
    ratio = pixels_per_row / cells_per_row
    dwg = _draw_grid(cairo_ctx, lambda ctx, x, y: ctx.rectangle(x, y, ratio, ratio), cells_per_row, pixels_per_row)
    return dwg

def circleboard(cairo_ctx, cells_per_row=10, pixels_per_row=1024):
    ratio = pixels_per_row / cells_per_row
    half_ratio = ratio / 2;
    dwg = _draw_grid(cairo_ctx, lambda ctx, x, y: ctx.arc(x + half_ratio,y + half_ratio, (half_ratio - 0.1 * ratio), 0, 2*pi), cells_per_row, pixels_per_row)
    return dwg

run()
