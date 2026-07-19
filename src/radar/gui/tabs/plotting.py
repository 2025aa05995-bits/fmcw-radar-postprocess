"""Shared matplotlib embedding and RX selection helpers for measurement tabs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ..theme import COLORS, style_axes, style_figure, style_mpl_toolbar


class EmbeddedFigure:
    """Matplotlib figure embedded in a parent frame; grows with the window."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] = (8.0, 4.5),
        toolbar: bool = True,
        tight: bool = True,
    ) -> None:
        # Avoid Figure(layout="tight") — it fights TkAgg resize and leaves
        # letterboxed white space. Use tight_layout() after draws instead.
        self._tight = tight
        self.fig = Figure(figsize=figsize, dpi=100)
        style_figure(self.fig)
        self.axes = self.fig.subplots(nrows=nrows, ncols=ncols)
        for ax in self.axes_list():
            style_axes(ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self._widget = self.canvas.get_tk_widget()
        try:
            self._widget.configure(bg=COLORS["plot_bg"], highlightthickness=0)
        except tk.TclError:
            pass
        self._widget.pack(side="top", fill="both", expand=True)
        self._widget.bind("<Configure>", self._on_configure, add="+")
        self._resize_after: str | None = None
        self.toolbar: NavigationToolbar2Tk | None = None
        if toolbar:
            self.toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=True)
            self.toolbar.update()
            style_mpl_toolbar(self.toolbar)

    def _on_configure(self, event: tk.Event) -> None:
        """Keep the figure size matched to the canvas widget."""
        if event.widget is not self._widget:
            return
        w, h = int(event.width), int(event.height)
        if w < 20 or h < 20:
            return
        dpi = float(self.fig.get_dpi())
        new_w, new_h = w / dpi, h / dpi
        cur_w, cur_h = self.fig.get_size_inches()
        if abs(cur_w - new_w) < 0.05 and abs(cur_h - new_h) < 0.05:
            return
        if self._resize_after is not None:
            try:
                self._widget.after_cancel(self._resize_after)
            except tk.TclError:
                pass
        self._resize_after = self._widget.after(30, lambda: self._apply_size(new_w, new_h))

    def _apply_size(self, width_in: float, height_in: float) -> None:
        self._resize_after = None
        # forward=False: update figure only; TkAgg owns the widget size.
        self.fig.set_size_inches(width_in, height_in, forward=False)
        if self._tight:
            try:
                self.fig.tight_layout()
            except Exception:  # noqa: BLE001
                pass
        self.canvas.draw_idle()

    def draw_idle(self) -> None:
        """Schedule a redraw without blocking the Tk event loop."""
        style_figure(self.fig)
        for ax in self.axes_list():
            style_axes(ax)
        if self._tight:
            try:
                self.fig.tight_layout()
            except Exception:  # noqa: BLE001
                pass
        self.canvas.draw_idle()

    def axes_list(self) -> list:
        """Return axes as a flat list."""
        if hasattr(self.axes, "flat"):
            return list(self.axes.flat)
        return [self.axes]


class RxChannelSelector:
    """
    Multi-select RX dropdown (checkable menu, up to ``max_rx`` channels).

    Call ``sync(num_rx)`` when the live cube size is known. Use ``selected()``
    for the checked indices. Defaults to RX 0 only.
    """

    def __init__(
        self,
        parent: ttk.Frame,
        *,
        max_rx: int = 24,
        default_all: bool = False,  # kept for call-site compat; ignored
        label: str = "RX:",
    ) -> None:
        from ..theme import COLORS

        self._max_rx = max(1, int(max_rx))
        self._num_rx = 0
        self._vars: list[tk.BooleanVar] = []
        self._colors = COLORS

        self.frame = ttk.Frame(parent)
        self.frame.pack(side="left", padx=(8, 0))
        ttk.Label(self.frame, text=label).pack(side="left")

        self._btn = ttk.Menubutton(self.frame, text="RX 0", direction="below", width=12)
        self._btn.pack(side="left", padx=(4, 0))
        self._menu = tk.Menu(
            self._btn,
            tearoff=0,
            bg=COLORS["surface_alt"],
            fg=COLORS["text"],
            activebackground=COLORS["elevated"],
            activeforeground=COLORS["text"],
            selectcolor=COLORS["accent"],
            bd=1,
            relief="flat",
        )
        self._btn["menu"] = self._menu

    def _rebuild_menu(self) -> None:
        """Rebuild the checkable RX menu for the current channel count."""
        self._menu.delete(0, tk.END)
        self._vars = []
        if self._num_rx <= 0:
            self._menu.add_command(label="(no RX)", state="disabled")
            self._refresh_button_text()
            return

        self._menu.add_command(label="All", command=self.select_all)
        self._menu.add_command(label="None", command=self.select_none)
        self._menu.add_separator()

        for rx in range(self._num_rx):
            # Default: only RX 0 checked when first building / after reset.
            var = tk.BooleanVar(value=(rx == 0))
            self._vars.append(var)
            self._menu.add_checkbutton(
                label=f"RX {rx}",
                variable=var,
                command=self._refresh_button_text,
            )
        self._refresh_button_text()

    def _refresh_button_text(self) -> None:
        """Update the menubutton summary from the current selection."""
        sel = self.selected()
        if not sel:
            text = "none"
        elif len(sel) == self._num_rx and self._num_rx > 1:
            text = f"all ({self._num_rx})"
        elif len(sel) <= 3:
            text = ",".join(str(i) for i in sel)
        else:
            text = f"{len(sel)} ch"
        self._btn.configure(text=text)

    def sync(self, num_rx: int) -> None:
        """Refresh menu options for the live RX count; keep prior picks when possible."""
        num_rx = max(0, min(int(num_rx), self._max_rx))
        if num_rx == self._num_rx and self._vars:
            return
        prev = set(self.selected()) if self._vars else {0}
        self._num_rx = num_rx
        self._rebuild_menu()
        if num_rx <= 0:
            return
        # Restore previous selection intersected with available channels.
        keep = sorted(i for i in prev if i < num_rx)
        if not keep:
            keep = [0]
        for i, var in enumerate(self._vars):
            var.set(i in keep)
        self._refresh_button_text()

    def selected(self) -> list[int]:
        """Return checked RX indices (may be empty)."""
        return [i for i, var in enumerate(self._vars[: self._num_rx]) if var.get()]

    def select_all(self) -> None:
        for var in self._vars[: self._num_rx]:
            var.set(True)
        self._refresh_button_text()

    def select_none(self) -> None:
        for var in self._vars[: self._num_rx]:
            var.set(False)
        self._refresh_button_text()

    def label(self) -> str:
        """Short summary for plot titles, e.g. ``RX 0,2`` or ``all RX``."""
        sel = self.selected()
        if not sel:
            return "no RX"
        if len(sel) == self._num_rx and self._num_rx > 1:
            return "all RX"
        return "RX " + ",".join(str(i) for i in sel)
