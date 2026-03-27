"""
app.py
------
AIS — Artifact Intelligence System
Upload an artifact photo, click Analyze, see similar artifacts.

First run: automatically downloads reference images and builds an index.
Subsequent runs: results in under a second.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import config
from ais.data.scraper import scrape_dataset
from ais.search import ArtifactSearcher

WIN_W, WIN_H = 960, 640
PREVIEW_SIZE = 260
THUMB_SIZE   = 150
RESULT_COLS  = 3
RESULT_ROWS  = 2
TOP_K        = RESULT_COLS * RESULT_ROWS


def _pil_to_tk(img: Image.Image, size: int) -> ImageTk.PhotoImage:
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), "#d9d9d9")
    x = (size - img.width)  // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return ImageTk.PhotoImage(canvas)


class AISApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AIS — Artifact Intelligence System")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)

        self._searcher: ArtifactSearcher | None = None
        self._uploaded_image: Image.Image | None = None
        self._thumb_refs: list[ImageTk.PhotoImage] = []
        self._preview_ref: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._load_searcher_async()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        menubar = tk.Menu(self)
        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="Rebuild Reference Database", command=self._confirm_rebuild)
        tools.add_separator()
        tools.add_command(label="Refine Model (trains on current data)", command=self._refine_model)
        menubar.add_cascade(label="Tools", menu=tools)
        self.configure(menu=menubar)

        main = tk.Frame(self)
        main.pack(fill="both", expand=True, padx=10, pady=8)

        self._build_left(main)
        tk.Frame(main, width=1, bg="gray70").pack(side="left", fill="y", padx=8)
        self._build_right(main)

        self._status_var = tk.StringVar(value="Starting up...")
        tk.Label(
            self, textvariable=self._status_var,
            relief="sunken", anchor="w", bd=1,
        ).pack(fill="x", side="bottom", ipady=2)

    def _build_left(self, parent: tk.Frame) -> None:
        left = tk.Frame(parent, width=290)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="Artifact Photo").pack(pady=(6, 4))

        self._preview_label = tk.Label(
            left,
            width=PREVIEW_SIZE, height=PREVIEW_SIZE,
            relief="sunken", bd=2,
            text="(click to choose photo)",
            fg="gray50",
            cursor="hand2",
            wraplength=200,
        )
        self._preview_label.pack()
        self._preview_label.bind("<Button-1>", lambda _: self._upload())

        tk.Frame(left, height=8).pack()
        tk.Button(left, text="Choose Photo...", command=self._upload, width=20).pack()
        tk.Frame(left, height=6).pack()

        self._analyze_btn = tk.Button(
            left, text="Analyze", command=self._analyze,
            width=20, state="disabled",
        )
        self._analyze_btn.pack()

    def _build_right(self, parent: tk.Frame) -> None:
        right = tk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(right, text="Similar Artifacts", anchor="w").pack(fill="x")
        tk.Frame(right, height=1, bg="gray70").pack(fill="x", pady=(2, 4))

        self._prediction_var = tk.StringVar()
        self._prediction_label = tk.Label(
            right, textvariable=self._prediction_var, anchor="w",
        )
        self._prediction_label.pack(fill="x", pady=(0, 4))

        self._results_frame = tk.Frame(right)
        self._results_frame.pack(fill="both", expand=True)

        self._show_placeholder()

    def _show_placeholder(self) -> None:
        for w in self._results_frame.winfo_children():
            w.destroy()
        tk.Label(
            self._results_frame,
            text="Upload a photo and click Analyze.",
            fg="gray50",
        ).place(relx=0.5, rely=0.4, anchor="center")

    # ── Startup ───────────────────────────────────────────────────────────────

    def _load_searcher_async(self) -> None:
        threading.Thread(target=self._load_searcher, daemon=True).start()

    def _load_searcher(self) -> None:
        """Load the searcher (pretrained model). No checkpoint needed."""
        try:
            self._set_status("Loading model...")
            self._searcher = ArtifactSearcher(
                embedding_dim=config.EMBEDDING_DIM,
                index_dir=config.CHECKPOINT_DIR,
            )
            if self._searcher.index_ready:
                n = len(self._searcher.image_paths)
                self._set_status(f"Ready — {n} artifacts in reference database.")
            else:
                self._set_status("Ready. Upload a photo and click Analyze.")
            self.after(0, self._refresh_analyze_btn)
        except Exception as exc:
            self._set_status(f"Startup error: {exc}")

    # ── Upload ────────────────────────────────────────────────────────────────

    def _upload(self) -> None:
        path = filedialog.askopenfilename(
            title="Select artifact photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        img = Image.open(path).convert("RGB")
        self._uploaded_image = img

        ref = _pil_to_tk(img, PREVIEW_SIZE)
        self._preview_ref = ref
        self._preview_label.configure(image=ref, text="")

        self._refresh_analyze_btn()
        self._set_status("Photo loaded. Click Analyze.")

    def _refresh_analyze_btn(self) -> None:
        ready = self._uploaded_image is not None and self._searcher is not None
        self._analyze_btn.configure(state="normal" if ready else "disabled")

    # ── Analyze ───────────────────────────────────────────────────────────────

    def _analyze(self) -> None:
        if self._uploaded_image is None or self._searcher is None:
            return

        self._analyze_btn.configure(state="disabled", text="Working...")

        if not self._searcher.index_ready:
            # First run — need to scrape + index before searching
            threading.Thread(target=self._first_run_then_search, daemon=True).start()
        else:
            threading.Thread(target=self._run_search, daemon=True).start()

    def _first_run_then_search(self) -> None:
        """Scrape → index → search on first Analyze (no training — DINOv2 zero-shot is better with small data)."""
        try:
            data_dir = Path(config.DATA_DIR)

            # ── Step 1: scrape ────────────────────────────────────────────────
            self._set_status("First run: downloading reference images (~2 min)...")
            scrape_dataset(
                classes=config.QUICK_CLASSES,
                out_dir=data_dir,
                n_images=config.QUICK_N_IMAGES,
                val_split=0.2,
                on_progress=self._set_status,
            )

            # ── Step 2: index with raw DINOv2 features ────────────────────────
            self._set_status("Building search index...")
            n = self._searcher.build_index(data_dir, on_progress=self._set_status)
            self._set_status(f"Index built — {n} reference artifacts.")

            # ── Step 3: search ────────────────────────────────────────────────
            self._run_search()

        except Exception as exc:
            self._set_status(f"Setup failed: {exc}")
            self.after(0, lambda: self._analyze_btn.configure(
                state="normal", text="Analyze"
            ))

    def _run_search(self) -> None:
        try:
            self._set_status("Analyzing...")
            results = self._searcher.search(self._uploaded_image, top_k=TOP_K)
            self.after(0, lambda: self._show_results(results))
        except Exception as exc:
            self._set_status(f"Analysis failed: {exc}")
        finally:
            self.after(0, lambda: self._analyze_btn.configure(
                state="normal", text="Analyze"
            ))

    # ── Results ───────────────────────────────────────────────────────────────

    def _show_results(self, results: list[dict]) -> None:
        for w in self._results_frame.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        if not results:
            tk.Label(self._results_frame, text="No results found.").pack()
            return

        top_class  = results[0]["predicted"]
        confidence = results[0]["confidence"]
        self._prediction_var.set(
            f"Most likely: {top_class.replace('_', ' ').title()}    ({confidence})"
        )

        grid = tk.Frame(self._results_frame)
        grid.pack(fill="both", expand=True)

        for i, res in enumerate(results):
            row, col = divmod(i, RESULT_COLS)
            cell = tk.Frame(grid, bd=1, relief="sunken", padx=4, pady=4)
            cell.grid(row=row, column=col, padx=6, pady=6)

            try:
                img = Image.open(res["path"]).convert("RGB")
                ref = _pil_to_tk(img, THUMB_SIZE)
            except Exception:
                ref = ImageTk.PhotoImage(Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), "#d9d9d9"))
            self._thumb_refs.append(ref)

            tk.Label(cell, image=ref).pack()
            tk.Label(cell, text=res["class"].replace("_", " ").title()).pack()
            pct = max(0, int(res["score"] * 100))
            tk.Label(cell, text=f"{pct}% match", fg="gray50").pack()

        self._set_status(
            f"Done — most likely {top_class.replace('_', ' ').title()}. "
            f"{len(results)} similar artifacts shown."
        )

    # ── Rebuild ───────────────────────────────────────────────────────────────

    def _confirm_rebuild(self) -> None:
        if messagebox.askyesno(
            "Rebuild Reference Database",
            "This will re-download all reference images and rebuild the index.\n"
            "Continue?",
        ):
            if self._searcher:
                # Clear the index so next Analyze triggers a full rebuild
                if self._searcher.index_path.exists():
                    self._searcher.index_path.unlink()
                self._searcher.embeddings  = None
                self._searcher.image_paths = []
                self._searcher.class_names = []
            self._set_status("Index cleared. Click Analyze to rebuild.")

    def _refine_model(self) -> None:
        """Train domain adaptation layers on current data, then rebuild index."""
        if self._searcher is None:
            messagebox.showinfo("AIS", "Model not loaded yet.")
            return
        data_dir = Path(config.DATA_DIR)
        if not (data_dir / "train").exists():
            messagebox.showerror("AIS", "No training data found. Run Analyze first.")
            return
        if not messagebox.askyesno(
            "Refine Model",
            "This will train the model on your current reference images (~3 min on CPU).\n"
            "Use this after collecting more artifact photos for better accuracy.\nContinue?",
        ):
            return

        def _run():
            try:
                self._set_status("Refining model...")
                self._searcher.train(
                    data_dir=data_dir,
                    epochs=config.EPOCHS,
                    batch_size=config.BATCH_SIZE,
                    lr=config.LEARNING_RATE,
                    on_progress=self._set_status,
                )
                self._set_status("Rebuilding index with refined model...")
                n = self._searcher.build_index(data_dir, on_progress=self._set_status)
                self.after(0, self._enable_search)
                self._set_status(f"Model refined — {n} artifacts re-indexed.")
            except Exception as exc:
                self._set_status(f"Refinement failed: {exc}")

        threading.Thread(target=_run, daemon=True).start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self.after(0, lambda: self._status_var.set(f" {msg}"))


if __name__ == "__main__":
    app = AISApp()
    app.mainloop()
