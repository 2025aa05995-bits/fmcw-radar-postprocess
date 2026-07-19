"""Dark UI theme for the live FMCW radar GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

# Slate instrument palette — deep charcoal with copper accent (no purple glow).
COLORS = {
    "bg": "#12141a",
    "surface": "#1a1d27",
    "surface_alt": "#222632",
    "elevated": "#2a3040",
    "border": "#3a4154",
    "text": "#e8eaef",
    "muted": "#9aa3b5",
    "accent": "#d4a574",
    "accent_hi": "#e8c49a",
    "accent_dim": "#8a6a45",
    "danger": "#d47a7a",
    "ok": "#6dbf8c",
    "plot_bg": "#0e1016",
    "plot_panel": "#141820",
    "grid": "#2a3142",
    "spine": "#4a5568",
    "trough": "#2a3040",
    "slider": "#d4a574",
}

# Trace cycle tuned for dark backgrounds.
_PLOT_CYCLE = [
    "#6ec6ff",
    "#f0b060",
    "#7dcea0",
    "#e07a7a",
    "#c39bd3",
    "#76d7c4",
    "#f5b7b1",
    "#aed6f1",
]


def apply_dark_theme(root: tk.Tk) -> ttk.Style:
    """
    Apply dark clam-based ttk styles and Tk option defaults to ``root``.

    Returns the configured ``ttk.Style`` for further tweaks.
    """
    root.configure(bg=COLORS["bg"])
    try:
        root.option_add("*tearOff", False)
    except tk.TclError:
        pass

    # Classic Tk widgets (Scale, Frame, Label used outside ttk).
    for pattern, value in (
        ("*Background", COLORS["bg"]),
        ("*Foreground", COLORS["text"]),
        ("*selectBackground", COLORS["accent_dim"]),
        ("*selectForeground", COLORS["text"]),
        ("*Frame.background", COLORS["bg"]),
        ("*Label.background", COLORS["bg"]),
        ("*Label.foreground", COLORS["text"]),
        ("*Canvas.background", COLORS["plot_bg"]),
        ("*Scale.background", COLORS["surface"]),
        ("*Scale.foreground", COLORS["text"]),
        ("*Scale.troughColor", COLORS["trough"]),
        ("*Scale.activeBackground", COLORS["accent"]),
        ("*Scale.highlightBackground", COLORS["bg"]),
        ("*Scale.highlightThickness", 0),
        ("*Entry.background", COLORS["surface_alt"]),
        ("*Entry.foreground", COLORS["text"]),
        ("*Entry.insertBackground", COLORS["accent"]),
        ("*Listbox.background", COLORS["surface_alt"]),
        ("*Listbox.foreground", COLORS["text"]),
    ):
        try:
            root.option_add(pattern, value)
        except tk.TclError:
            pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    bg = COLORS["bg"]
    surface = COLORS["surface"]
    surface_alt = COLORS["surface_alt"]
    elevated = COLORS["elevated"]
    border = COLORS["border"]
    text = COLORS["text"]
    muted = COLORS["muted"]
    accent = COLORS["accent"]
    accent_hi = COLORS["accent_hi"]

    style.configure(".", background=bg, foreground=text, fieldbackground=surface_alt)
    style.configure("TFrame", background=bg)
    style.configure("Card.TFrame", background=surface, relief="flat")
    style.configure("Header.TFrame", background=surface)
    style.configure("Toolbar.TFrame", background=surface)

    style.configure("TLabel", background=bg, foreground=text)
    style.configure("Muted.TLabel", background=bg, foreground=muted)
    style.configure("Header.TLabel", background=surface, foreground=text, font=("Segoe UI", 14, "bold"))
    style.configure("Subheader.TLabel", background=surface, foreground=muted, font=("Segoe UI", 9))
    style.configure("HeaderMuted.TLabel", background=surface, foreground=muted, font=("Segoe UI", 9))
    style.configure("Status.TLabel", background=surface, foreground=muted, font=("Segoe UI", 9))
    style.configure("Brand.TLabel", background=surface, foreground=accent, font=("Segoe UI Semibold", 15, "bold"))

    style.configure(
        "TButton",
        background=elevated,
        foreground=text,
        bordercolor=border,
        lightcolor=elevated,
        darkcolor=elevated,
        focuscolor=accent,
        padding=(10, 5),
    )
    style.map(
        "TButton",
        background=[("active", border), ("disabled", surface)],
        foreground=[("disabled", muted)],
    )
    style.configure(
        "Accent.TButton",
        background=accent,
        foreground="#1a1208",
        bordercolor=accent_hi,
        lightcolor=accent_hi,
        darkcolor=accent,
        padding=(12, 5),
    )
    style.map(
        "Accent.TButton",
        background=[("active", accent_hi), ("disabled", elevated)],
        foreground=[("disabled", muted)],
    )
    style.configure(
        "Ghost.TButton",
        background=surface,
        foreground=text,
        bordercolor=border,
        padding=(10, 5),
    )

    style.configure(
        "TEntry",
        fieldbackground=surface_alt,
        foreground=text,
        insertcolor=accent,
        bordercolor=border,
        lightcolor=border,
        darkcolor=border,
        padding=4,
    )
    style.map("TEntry", fieldbackground=[("focus", elevated)])

    style.configure(
        "TCombobox",
        fieldbackground=surface_alt,
        background=elevated,
        foreground=text,
        arrowcolor=accent,
        bordercolor=border,
        lightcolor=border,
        darkcolor=border,
        padding=4,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", surface_alt)],
        foreground=[("readonly", text)],
        background=[("active", border)],
    )
    root.option_add("*TCombobox*Listbox.background", surface_alt)
    root.option_add("*TCombobox*Listbox.foreground", text)
    root.option_add("*TCombobox*Listbox.selectBackground", COLORS["accent_dim"])
    root.option_add("*TCombobox*Listbox.selectForeground", text)

    style.configure(
        "TCheckbutton",
        background=bg,
        foreground=text,
        indicatorcolor=surface_alt,
        indicatormargin=3,
    )
    style.map(
        "TCheckbutton",
        background=[("active", bg)],
        indicatorcolor=[("selected", accent), ("!selected", surface_alt)],
    )

    style.configure(
        "TRadiobutton",
        background=bg,
        foreground=text,
        indicatorcolor=surface_alt,
    )
    style.map(
        "TRadiobutton",
        background=[("active", bg)],
        indicatorcolor=[("selected", accent), ("!selected", surface_alt)],
    )

    style.configure(
        "TNotebook",
        background=bg,
        borderwidth=0,
        tabmargins=(6, 6, 6, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=surface,
        foreground=muted,
        padding=(14, 7),
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", elevated), ("active", surface_alt)],
        foreground=[("selected", accent_hi), ("active", text)],
    )

    style.configure("TSeparator", background=border)
    style.configure(
        "TSpinbox",
        fieldbackground=surface_alt,
        foreground=text,
        background=elevated,
        arrowcolor=accent,
        bordercolor=border,
        padding=3,
    )

    style.configure(
        "Vertical.TScrollbar",
        background=elevated,
        troughcolor=surface,
        bordercolor=surface,
        arrowcolor=muted,
    )

    apply_matplotlib_rc()
    return style


def apply_matplotlib_rc() -> None:
    """Set global matplotlib rcParams for dark plots."""
    try:
        import matplotlib as mpl
    except ImportError:
        return
    mpl.rcParams.update(
        {
            "figure.facecolor": COLORS["plot_bg"],
            "figure.edgecolor": COLORS["plot_bg"],
            "axes.facecolor": COLORS["plot_panel"],
            "axes.edgecolor": COLORS["spine"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.85,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["text"],
            "legend.facecolor": COLORS["surface"],
            "legend.edgecolor": COLORS["border"],
            "legend.fontsize": 7,
            "legend.labelcolor": COLORS["text"],
            "axes.prop_cycle": mpl.cycler(color=_PLOT_CYCLE),
            "savefig.facecolor": COLORS["plot_bg"],
            "font.size": 9,
        }
    )


def style_figure(fig: Any) -> None:
    """Apply dark colors to an existing matplotlib Figure."""
    fig.patch.set_facecolor(COLORS["plot_bg"])


def style_axes(ax: Any) -> None:
    """Apply dark colors to a matplotlib Axes (safe after ``clear()``)."""
    ax.set_facecolor(COLORS["plot_panel"])
    ax.tick_params(colors=COLORS["muted"], which="both")
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["spine"])
    ax.grid(True, color=COLORS["grid"], alpha=0.85, linewidth=0.6)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor(COLORS["surface"])
        legend.get_frame().set_edgecolor(COLORS["border"])
        for text in legend.get_texts():
            text.set_color(COLORS["text"])


def style_tk_scale(scale: tk.Scale, *, bordered: bool = True) -> None:
    """Restyle a classic ``tk.Scale`` for the dark theme."""
    scale.configure(
        bg=COLORS["surface"],
        fg=COLORS["text"],
        troughcolor=COLORS["trough"],
        activebackground=COLORS["accent"],
        highlightthickness=1 if bordered else 0,
        highlightbackground=COLORS["border"],
        highlightcolor=COLORS["accent_hi"],
        bd=0,
        relief="flat",
    )


def bordered_panel(parent: tk.Misc, *, padx: int = 6, pady: int = 4) -> tk.Frame:
    """
    Light-bordered panel for grouping a slider + labels on the dark theme.
    """
    outer = tk.Frame(
        parent,
        bg=COLORS["border"],
        highlightthickness=0,
        bd=0,
    )
    inner = tk.Frame(
        outer,
        bg=COLORS["surface"],
        highlightthickness=0,
        bd=0,
        padx=padx,
        pady=pady,
    )
    # 1px border via outer/inner padding trick
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    outer._content = inner  # type: ignore[attr-defined]
    return outer


def panel_content(panel: tk.Frame) -> tk.Frame:
    """Return the inner content frame of a ``bordered_panel``."""
    return getattr(panel, "_content", panel)


def style_mpl_toolbar(toolbar: Any) -> None:
    """Darken a Matplotlib ``NavigationToolbar2Tk`` strip."""
    bg = COLORS["surface"]
    try:
        toolbar.configure(background=bg)
    except tk.TclError:
        pass
    for child in toolbar.winfo_children():
        try:
            child.configure(background=bg)
        except tk.TclError:
            pass
        try:
            child.configure(foreground=COLORS["text"])
        except tk.TclError:
            pass
