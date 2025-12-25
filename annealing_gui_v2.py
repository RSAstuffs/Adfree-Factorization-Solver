#!/usr/bin/env python3
"""
Advanced GUI for Quantum Annealing Factorization
Version 2.3 - MLClauseLearner Neural Network (Exclusive Decision Maker)

Features:
- Real-time progress visualization
- Learning statistics dashboard
- Elite population viewer
- Tabu/Nogood learning display
- Bad bit combination tracking
- Convergence graph
- Dark/Light theme support
- Better parameter organization
- NEW: Policy Network Training & Inference
- NEW: Carry-Aware Propagation Stats
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import sys
import os
import json
import time
import hashlib
from typing import Optional, Dict, List
import math


def get_safe_state_filename(n: int) -> str:
    """Generate a safe state filename that won't exceed filesystem limits.
    
    For small numbers (< 100 digits), use the number directly.
    For large numbers, use a hash with bit length prefix.
    """
    n_str = str(n)
    if len(n_str) <= 100:
        return f"state_{n_str}.json"
    else:
        # Use hash for large numbers
        n_hash = hashlib.md5(n_str.encode()).hexdigest()[:16]
        bit_length = n.bit_length()
        return f"state_{bit_length}bit_{n_hash}.json"

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ModernStyle:
    """Modern color scheme and styling."""
    
    # Dark theme colors
    DARK = {
        'bg': '#1e1e2e',
        'bg_secondary': '#2d2d3d',
        'fg': '#cdd6f4',
        'fg_dim': '#6c7086',
        'accent': '#89b4fa',
        'success': '#a6e3a1',
        'warning': '#f9e2af',
        'error': '#f38ba8',
        'border': '#45475a',
        'button': '#313244',
        'button_hover': '#45475a',
    }
    
    # Light theme colors
    LIGHT = {
        'bg': '#eff1f5',
        'bg_secondary': '#e6e9ef',
        'fg': '#4c4f69',
        'fg_dim': '#8c8fa1',
        'accent': '#1e66f5',
        'success': '#40a02b',
        'warning': '#df8e1d',
        'error': '#d20f39',
        'border': '#ccd0da',
        'button': '#dce0e8',
        'button_hover': '#ccd0da',
    }
    
    current = DARK  # Default theme


class FactorizationGUI:
    """Main GUI application for quantum annealing factorization."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Quantum Annealing Factorization v2.3 - Neural Network Exclusive")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # State
        self.annealer = None
        self.annealing_thread = None
        self.stop_flag = False
        self.is_running = False
        self.current_theme = 'dark'
        
        # Statistics
        self.restart_count = 0
        self.best_diff = float('inf')
        self.history = []
        self.start_time = None
        
        # Apply theme
        self.apply_theme()
        
        # Build UI
        self.setup_ui()
        
        # Load default presets
        self.load_preset_2021()
    
    def apply_theme(self):
        """Apply the current color theme."""
        colors = ModernStyle.DARK if self.current_theme == 'dark' else ModernStyle.LIGHT
        ModernStyle.current = colors
        
        self.root.configure(bg=colors['bg'])
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure all widget styles
        style.configure('TFrame', background=colors['bg'])
        style.configure('TLabel', background=colors['bg'], foreground=colors['fg'])
        style.configure('TButton', background=colors['button'], foreground=colors['fg'])
        style.configure('TEntry', fieldbackground=colors['bg_secondary'], foreground=colors['fg'])
        style.configure('TCheckbutton', background=colors['bg'], foreground=colors['fg'])
        style.configure('TLabelframe', background=colors['bg'], foreground=colors['accent'])
        style.configure('TLabelframe.Label', background=colors['bg'], foreground=colors['accent'])
        style.configure('TNotebook', background=colors['bg'])
        style.configure('TNotebook.Tab', background=colors['button'], foreground=colors['fg'])
        
        # Custom styles
        style.configure('Accent.TButton', background=colors['accent'], foreground='white')
        style.configure('Success.TLabel', foreground=colors['success'])
        style.configure('Warning.TLabel', foreground=colors['warning'])
        style.configure('Error.TLabel', foreground=colors['error'])
        style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Title.TLabel', font=('Helvetica', 18, 'bold'), foreground=colors['accent'])
        style.configure('Stats.TLabel', font=('Consolas', 11))
    
    def setup_ui(self):
        """Build the main user interface."""
        colors = ModernStyle.current
        
        # Main container
        main_container = ttk.Frame(self.root, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self.create_header(main_container)
        
        # Content area with notebook tabs
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel - Parameters (wider to fit all controls)
        left_panel = ttk.Frame(content_frame, width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self.create_parameters_panel(left_panel)
        
        # Right panel - Tabs for output/stats/elite
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Live Output
        output_frame = ttk.Frame(self.notebook)
        self.notebook.add(output_frame, text="  📝 Live Output  ")
        self.create_output_tab(output_frame)
        
        # Tab 2: Statistics
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="  📊 Statistics  ")
        self.create_stats_tab(stats_frame)
        
        # Tab 3: Elite Population
        elite_frame = ttk.Frame(self.notebook)
        self.notebook.add(elite_frame, text="  🏆 Elite Solutions  ")
        self.create_elite_tab(elite_frame)
        
        # Tab 4: Learning
        learning_frame = ttk.Frame(self.notebook)
        self.notebook.add(learning_frame, text="  🧠 Learning  ")
        self.create_learning_tab(learning_frame)
        
        # Tab 5: Policy Network (NEW)
        policy_frame = ttk.Frame(self.notebook)
        self.notebook.add(policy_frame, text="  🤖 Policy Net  ")
        self.create_policy_tab(policy_frame)
        
        # Bottom status bar
        self.create_status_bar(main_container)
    
    def create_header(self, parent):
        """Create the header section."""
        colors = ModernStyle.current
        
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(header_frame, text="⚛️ Quantum Annealing Factorization",
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Theme toggle
        self.theme_btn = ttk.Button(header_frame, text="🌙 Dark" if self.current_theme == 'dark' else "☀️ Light",
                                    command=self.toggle_theme, width=10)
        self.theme_btn.pack(side=tk.RIGHT, padx=5)
        
        # Help button
        help_btn = ttk.Button(header_frame, text="❓ Help", command=self.show_help, width=10)
        help_btn.pack(side=tk.RIGHT, padx=5)
    
    def create_parameters_panel(self, parent):
        """Create the parameters input panel."""
        colors = ModernStyle.current
        
        # Scrollable frame for parameters
        canvas = tk.Canvas(parent, bg=colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=360)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        def _on_mousewheel_linux(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/Mac
        canvas.bind_all("<Button-4>", _on_mousewheel_linux)  # Linux scroll up
        canvas.bind_all("<Button-5>", _on_mousewheel_linux)  # Linux scroll down
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === Target Number ===
        target_frame = ttk.LabelFrame(scroll_frame, text="🎯 Target Number", padding=10)
        target_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(target_frame, text="N (Number to factor):").pack(anchor=tk.W)
        self.n_var = tk.StringVar(value="2021")
        n_entry = ttk.Entry(target_frame, textvariable=self.n_var, width=40)
        n_entry.pack(fill=tk.X, pady=(2, 5))
        
        # Bind N entry to auto-calculate pairs
        self.n_var.trace_add("write", self._on_n_changed)
        
        # Quick presets
        preset_frame = ttk.Frame(target_frame)
        preset_frame.pack(fill=tk.X)
        ttk.Button(preset_frame, text="2021", command=self.load_preset_2021, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="143", command=self.load_preset_143, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="15", command=self.load_preset_15, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Custom", command=self.load_custom, width=8).pack(side=tk.LEFT, padx=2)
        
        # === Annealing Parameters ===
        anneal_frame = ttk.LabelFrame(scroll_frame, text="🔥 Annealing Parameters", padding=10)
        anneal_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Number of triangle pairs - AUTO SCALING
        pairs_label_frame = ttk.Frame(anneal_frame)
        pairs_label_frame.pack(fill=tk.X)
        ttk.Label(pairs_label_frame, text="Triangle Pairs:").pack(side=tk.LEFT)
        self.auto_pairs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pairs_label_frame, text="Auto", variable=self.auto_pairs_var, 
                       command=self._on_auto_pairs_changed).pack(side=tk.LEFT, padx=10)
        
        self.pairs_var = tk.StringVar(value="14")
        pairs_frame = ttk.Frame(anneal_frame)
        pairs_frame.pack(fill=tk.X, pady=(2, 5))
        self.pairs_entry = ttk.Entry(pairs_frame, textvariable=self.pairs_var, width=10, state='disabled')
        self.pairs_entry.pack(side=tk.LEFT)
        self.pairs_info_label = ttk.Label(pairs_frame, text="(7 for p, 7 for q)", style='Stats.TLabel')
        self.pairs_info_label.pack(side=tk.LEFT, padx=10)
        
        # Steps per restart
        ttk.Label(anneal_frame, text="Steps per Restart:").pack(anchor=tk.W)
        self.steps_var = tk.StringVar(value="80")
        ttk.Entry(anneal_frame, textvariable=self.steps_var, width=10).pack(anchor=tk.W, pady=(2, 5))
        
        # Reads per step
        ttk.Label(anneal_frame, text="Reads per Step:").pack(anchor=tk.W)
        self.reads_var = tk.StringVar(value="15")
        ttk.Entry(anneal_frame, textvariable=self.reads_var, width=10).pack(anchor=tk.W, pady=(2, 5))
        
        # === Temperature Schedule ===
        temp_frame = ttk.LabelFrame(scroll_frame, text="🌡️ Temperature Schedule", padding=10)
        temp_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Auto-scale checkbox
        self.auto_temp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(temp_frame, text="Auto-scale by N size (recommended)", 
                       variable=self.auto_temp_var,
                       command=self._toggle_temp_entries).pack(anchor=tk.W, pady=(0, 5))
        
        # Manual temp entries (disabled by default when auto-scale is on)
        self.temp_entries_frame = ttk.Frame(temp_frame)
        self.temp_entries_frame.pack(fill=tk.X)
        
        ttk.Label(self.temp_entries_frame, text="Initial Temperature:").pack(anchor=tk.W)
        self.init_temp_var = tk.StringVar(value="Auto")
        self.init_temp_entry = ttk.Entry(self.temp_entries_frame, textvariable=self.init_temp_var, width=15, state='disabled')
        self.init_temp_entry.pack(anchor=tk.W, pady=(2, 5))
        
        ttk.Label(self.temp_entries_frame, text="Final Temperature:").pack(anchor=tk.W)
        self.final_temp_var = tk.StringVar(value="Auto")
        self.final_temp_entry = ttk.Entry(self.temp_entries_frame, textvariable=self.final_temp_var, width=15, state='disabled')
        self.final_temp_entry.pack(anchor=tk.W, pady=(2, 5))
        
        # Info label about auto-scaling
        self.temp_info_label = ttk.Label(temp_frame, text="📐 Temp scales: 100×(1+log₂(N_bits))", foreground='gray')
        self.temp_info_label.pack(anchor=tk.W, pady=(2, 0))
        
        # === Convergence Settings ===
        conv_frame = ttk.LabelFrame(scroll_frame, text="🎯 Convergence Settings", padding=10)
        conv_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.converge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(conv_frame, text="Run until exact factorization", 
                       variable=self.converge_var).pack(anchor=tk.W)
        
        ttk.Label(conv_frame, text="Max Restarts (0 = unlimited):").pack(anchor=tk.W, pady=(5, 0))
        self.max_restarts_var = tk.StringVar(value="0")
        ttk.Entry(conv_frame, textvariable=self.max_restarts_var, width=10).pack(anchor=tk.W, pady=(2, 5))
        
        ttk.Label(conv_frame, text="State File:").pack(anchor=tk.W)
        state_frame = ttk.Frame(conv_frame)
        state_frame.pack(fill=tk.X, pady=(2, 5))
        self.state_file_var = tk.StringVar(value="annealing_state.json")
        ttk.Entry(state_frame, textvariable=self.state_file_var, width=25).pack(side=tk.LEFT)
        ttk.Button(state_frame, text="📁", command=self.browse_state_file, width=3).pack(side=tk.LEFT, padx=2)
        
        # === Neural Network (MLClauseLearner - Primary Decision Maker) ===
        nn_frame = ttk.LabelFrame(scroll_frame, text="🧠 Neural Network (MLClauseLearner)", padding=10)
        nn_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Neural network status - this is saved/loaded from state file
        ttk.Label(nn_frame, text="The neural network makes ALL bit flip decisions.").pack(anchor=tk.W)
        ttk.Label(nn_frame, text="Knowledge is saved/loaded from State File above.").pack(anchor=tk.W)
        self.nn_status_var = tk.StringVar(value="Status: Not initialized")
        self.nn_status_label = ttk.Label(nn_frame, textvariable=self.nn_status_var, 
                                         font=('Consolas', 9), foreground='#00aa00')
        self.nn_status_label.pack(anchor=tk.W, pady=(5, 0))
        
        self.nn_samples_var = tk.StringVar(value="Training samples: 0")
        ttk.Label(nn_frame, textvariable=self.nn_samples_var).pack(anchor=tk.W)
        
        self.nn_patterns_var = tk.StringVar(value="Best patterns: 0")
        ttk.Label(nn_frame, textvariable=self.nn_patterns_var).pack(anchor=tk.W)
        
        # === Bit Selection Strategy ===
        strategy_frame = ttk.LabelFrame(scroll_frame, text="🎛️ Bit Selection Strategy", padding=10)
        strategy_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(strategy_frame, text="Control which methods select bits to flip:", 
                  font=('TkDefaultFont', 9)).pack(anchor=tk.W, pady=(0, 5))
        
        # Transformer % (LLM-style attention)
        transformer_row = ttk.Frame(strategy_frame)
        transformer_row.pack(fill=tk.X, pady=2)
        ttk.Label(transformer_row, text="🤖 Transformer (Attention):", width=22).pack(side=tk.LEFT)
        self.transformer_pct_var = tk.IntVar(value=30)
        self.transformer_scale = ttk.Scale(transformer_row, from_=0, to=100, 
                                           variable=self.transformer_pct_var,
                                           orient=tk.HORIZONTAL, length=120,
                                           command=self._on_strategy_change)
        self.transformer_scale.pack(side=tk.LEFT, padx=5)
        self.transformer_pct_label = ttk.Label(transformer_row, text="30%", width=5)
        self.transformer_pct_label.pack(side=tk.LEFT)
        
        # Hourglass Neural Network %
        hourglass_row = ttk.Frame(strategy_frame)
        hourglass_row.pack(fill=tk.X, pady=2)
        ttk.Label(hourglass_row, text="🧠 Hourglass Network:", width=22).pack(side=tk.LEFT)
        self.hourglass_pct_var = tk.IntVar(value=50)
        self.hourglass_scale = ttk.Scale(hourglass_row, from_=0, to=100,
                                         variable=self.hourglass_pct_var,
                                         orient=tk.HORIZONTAL, length=120,
                                         command=self._on_strategy_change)
        self.hourglass_scale.pack(side=tk.LEFT, padx=5)
        self.hourglass_pct_label = ttk.Label(hourglass_row, text="50%", width=5)
        self.hourglass_pct_label.pack(side=tk.LEFT)
        
        # Random Exploration %
        random_row = ttk.Frame(strategy_frame)
        random_row.pack(fill=tk.X, pady=2)
        ttk.Label(random_row, text="🎲 Random Exploration:", width=22).pack(side=tk.LEFT)
        self.random_pct_var = tk.IntVar(value=20)
        self.random_scale = ttk.Scale(random_row, from_=0, to=100,
                                      variable=self.random_pct_var,
                                      orient=tk.HORIZONTAL, length=120,
                                      command=self._on_strategy_change)
        self.random_scale.pack(side=tk.LEFT, padx=5)
        self.random_pct_label = ttk.Label(random_row, text="20%", width=5)
        self.random_pct_label.pack(side=tk.LEFT)
        
        # Total indicator and normalize button
        total_row = ttk.Frame(strategy_frame)
        total_row.pack(fill=tk.X, pady=(5, 2))
        self.strategy_total_var = tk.StringVar(value="Total: 100%")
        ttk.Label(total_row, textvariable=self.strategy_total_var, 
                  font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
        ttk.Button(total_row, text="⚖️ Normalize", command=self._normalize_strategy,
                   width=10).pack(side=tk.RIGHT)
        
        # Info label
        ttk.Label(strategy_frame, text="💡 Transformer learns bit correlations via attention", 
                  foreground='gray', font=('TkDefaultFont', 8)).pack(anchor=tk.W, pady=(2, 0))
        
        # === Metropolis Acceptance Settings ===
        metro_frame = ttk.LabelFrame(scroll_frame, text="🎰 Metropolis Acceptance", padding=10)
        metro_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(metro_frame, text="Controls how strictly uphill moves are rejected:", 
                  font=('TkDefaultFont', 9)).pack(anchor=tk.W, pady=(0, 5))
        
        # Minimum acceptance probability slider
        accept_row = ttk.Frame(metro_frame)
        accept_row.pack(fill=tk.X, pady=2)
        ttk.Label(accept_row, text="Min Accept Prob:", width=15).pack(side=tk.LEFT)
        self.metro_min_accept_var = tk.DoubleVar(value=0.05)
        self.metro_scale = ttk.Scale(accept_row, from_=0.0, to=0.30,
                                      variable=self.metro_min_accept_var,
                                      orient=tk.HORIZONTAL, length=100,
                                      command=self._on_metro_change)
        self.metro_scale.pack(side=tk.LEFT, padx=5)
        self.metro_pct_label = ttk.Label(accept_row, text="5%", width=5)
        self.metro_pct_label.pack(side=tk.LEFT)
        
        # Preset buttons
        metro_preset_row = ttk.Frame(metro_frame)
        metro_preset_row.pack(fill=tk.X, pady=(5, 2))
        ttk.Button(metro_preset_row, text="Strict (1%)", width=10,
                   command=lambda: self._set_metro_leniency(0.01)).pack(side=tk.LEFT, padx=2)
        ttk.Button(metro_preset_row, text="Normal (5%)", width=10,
                   command=lambda: self._set_metro_leniency(0.05)).pack(side=tk.LEFT, padx=2)
        ttk.Button(metro_preset_row, text="Lenient (15%)", width=10,
                   command=lambda: self._set_metro_leniency(0.15)).pack(side=tk.LEFT, padx=2)
        ttk.Button(metro_preset_row, text="Very Lenient (25%)", width=12,
                   command=lambda: self._set_metro_leniency(0.25)).pack(side=tk.LEFT, padx=2)
        
        # Info
        ttk.Label(metro_frame, text="💡 Higher = more exploration, better ML learning", 
                  foreground='gray', font=('TkDefaultFont', 8)).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(metro_frame, text="💡 Lower = stricter convergence, less exploration", 
                  foreground='gray', font=('TkDefaultFont', 8)).pack(anchor=tk.W)
        
        # === Legacy Policy Network (Optional - for initialization only) ===
        policy_frame = ttk.LabelFrame(scroll_frame, text="📦 Legacy PolicyNetwork (Init Only)", padding=10)
        policy_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.use_policy_var = tk.BooleanVar(value=False)  # Default OFF - MLClauseLearner is primary
        ttk.Checkbutton(policy_frame, text="Seed initial weights from PolicyNetwork (optional)", 
                       variable=self.use_policy_var).pack(anchor=tk.W)
        
        ttk.Label(policy_frame, text="Policy File:").pack(anchor=tk.W, pady=(5, 0))
        policy_file_frame = ttk.Frame(policy_frame)
        policy_file_frame.pack(fill=tk.X, pady=(2, 5))
        self.policy_file_var = tk.StringVar(value="policy_network.npz")
        ttk.Entry(policy_file_frame, textvariable=self.policy_file_var, width=20).pack(side=tk.LEFT)
        ttk.Button(policy_file_frame, text="📁", command=self.browse_policy_file, width=3).pack(side=tk.LEFT, padx=2)
        
        # Policy training button
        train_frame = ttk.Frame(policy_frame)
        train_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(train_frame, text="Episodes:").pack(side=tk.LEFT)
        self.train_episodes_var = tk.StringVar(value="500")
        ttk.Entry(train_frame, textvariable=self.train_episodes_var, width=8).pack(side=tk.LEFT, padx=5)
        self.train_policy_btn = ttk.Button(train_frame, text="🎓 Train", command=self.train_policy_network, width=8)
        self.train_policy_btn.pack(side=tk.LEFT, padx=5)
        
        # Policy status
        self.policy_status_var = tk.StringVar(value="Not loaded")
        self.policy_status_label = ttk.Label(policy_frame, textvariable=self.policy_status_var, 
                                             style='Stats.TLabel')
        self.policy_status_label.pack(anchor=tk.W, pady=(5, 0))
        
        # === Control Buttons ===
        control_frame = ttk.LabelFrame(scroll_frame, text="▶️ Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Top row: Start and Stop
        btn_row1 = ttk.Frame(control_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        
        self.start_btn = ttk.Button(btn_row1, text="▶️ Start", command=self.start_annealing, 
                                    style='Accent.TButton', width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        self.stop_btn = ttk.Button(btn_row1, text="⏹️ Stop", command=self.stop_annealing, 
                                   width=15, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        # Bottom row: Resume
        btn_row2 = ttk.Frame(control_frame)
        btn_row2.pack(fill=tk.X, pady=2)
        
        self.resume_btn = ttk.Button(btn_row2, text="⏯️ Resume from State", command=self.resume_annealing, width=32)
        self.resume_btn.pack(pady=2)
    
    def create_output_tab(self, parent):
        """Create the live output tab."""
        colors = ModernStyle.current
        
        # Output text with custom styling
        self.output_text = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, font=('Consolas', 10),
            bg=colors['bg_secondary'], fg=colors['fg'],
            insertbackground=colors['fg']
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure tags for colored output
        self.output_text.tag_configure('success', foreground=colors['success'])
        self.output_text.tag_configure('warning', foreground=colors['warning'])
        self.output_text.tag_configure('error', foreground=colors['error'])
        self.output_text.tag_configure('info', foreground=colors['accent'])
        self.output_text.tag_configure('dim', foreground=colors['fg_dim'])
        
        # Button bar
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="🗑️ Clear", command=self.clear_output).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="💾 Save Log", command=self.save_log).pack(side=tk.LEFT, padx=2)
        
        self.autoscroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(btn_frame, text="Auto-scroll", variable=self.autoscroll_var).pack(side=tk.RIGHT)
    
    def create_stats_tab(self, parent):
        """Create the statistics tab."""
        colors = ModernStyle.current
        
        # Stats grid
        stats_container = ttk.Frame(parent, padding=10)
        stats_container.pack(fill=tk.BOTH, expand=True)
        
        # Current Status
        status_frame = ttk.LabelFrame(stats_container, text="📈 Current Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)
        
        # Row 1
        ttk.Label(status_grid, text="Restarts:", style='Stats.TLabel').grid(row=0, column=0, sticky=tk.W, padx=5)
        self.stat_restarts = ttk.Label(status_grid, text="0", style='Stats.TLabel')
        self.stat_restarts.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(status_grid, text="Best Diff:", style='Stats.TLabel').grid(row=0, column=2, sticky=tk.W, padx=20)
        self.stat_best_diff = ttk.Label(status_grid, text="∞", style='Stats.TLabel')
        self.stat_best_diff.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Row 2
        ttk.Label(status_grid, text="Elapsed:", style='Stats.TLabel').grid(row=1, column=0, sticky=tk.W, padx=5)
        self.stat_elapsed = ttk.Label(status_grid, text="0:00", style='Stats.TLabel')
        self.stat_elapsed.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(status_grid, text="Temperature:", style='Stats.TLabel').grid(row=1, column=2, sticky=tk.W, padx=20)
        self.stat_temp = ttk.Label(status_grid, text="-", style='Stats.TLabel')
        self.stat_temp.grid(row=1, column=3, sticky=tk.W, padx=5)
        
        # Best Solution
        best_frame = ttk.LabelFrame(stats_container, text="🏆 Best Solution", padding=10)
        best_frame.pack(fill=tk.X, pady=5)
        
        self.stat_best_solution = ttk.Label(best_frame, text="No solution yet", 
                                            font=('Consolas', 14), style='Stats.TLabel')
        self.stat_best_solution.pack(anchor=tk.W)
        
        self.stat_best_product = ttk.Label(best_frame, text="", style='Stats.TLabel')
        self.stat_best_product.pack(anchor=tk.W)
        
        # Learning Stats
        learn_frame = ttk.LabelFrame(stats_container, text="🧠 Learning Statistics", padding=10)
        learn_frame.pack(fill=tk.X, pady=5)
        
        learn_grid = ttk.Frame(learn_frame)
        learn_grid.pack(fill=tk.X)
        
        # Row 0: Elite and Clauses
        ttk.Label(learn_grid, text="Elite Population:", style='Stats.TLabel').grid(row=0, column=0, sticky=tk.W, padx=5)
        self.stat_elite = ttk.Label(learn_grid, text="0", style='Stats.TLabel')
        self.stat_elite.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(learn_grid, text="Learned Clauses:", style='Stats.TLabel').grid(row=0, column=2, sticky=tk.W, padx=20)
        self.stat_clauses = ttk.Label(learn_grid, text="0", style='Stats.TLabel')
        self.stat_clauses.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Row 1: Correlations and Stuck
        ttk.Label(learn_grid, text="Bit Correlations:", style='Stats.TLabel').grid(row=1, column=0, sticky=tk.W, padx=5)
        self.stat_correlations = ttk.Label(learn_grid, text="0", style='Stats.TLabel')
        self.stat_correlations.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(learn_grid, text="Stuck Counter:", style='Stats.TLabel').grid(row=1, column=2, sticky=tk.W, padx=20)
        self.stat_stuck = ttk.Label(learn_grid, text="0", style='Stats.TLabel')
        self.stat_stuck.grid(row=1, column=3, sticky=tk.W, padx=5)
        
        # Row 2: Tabu and Nogoods (NEW)
        ttk.Label(learn_grid, text="Tabu List:", style='Stats.TLabel').grid(row=2, column=0, sticky=tk.W, padx=5)
        self.stat_tabu = ttk.Label(learn_grid, text="0", style='Stats.TLabel')
        self.stat_tabu.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(learn_grid, text="Nogood Patterns:", style='Stats.TLabel').grid(row=2, column=2, sticky=tk.W, padx=20)
        self.stat_nogoods = ttk.Label(learn_grid, text="0", style='Stats.TLabel')
        self.stat_nogoods.grid(row=2, column=3, sticky=tk.W, padx=5)
        
        # Row 3: Bad Combos and Thresholds (NEW)
        ttk.Label(learn_grid, text="Bad Bit Combos:", style='Stats.TLabel').grid(row=3, column=0, sticky=tk.W, padx=5)
        self.stat_bad_combos = ttk.Label(learn_grid, text="0", style='Stats.TLabel')
        self.stat_bad_combos.grid(row=3, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(learn_grid, text="Thresholds:", style='Stats.TLabel').grid(row=3, column=2, sticky=tk.W, padx=20)
        self.stat_threshold = ttk.Label(learn_grid, text="-", style='Stats.TLabel')
        self.stat_threshold.grid(row=3, column=3, sticky=tk.W, padx=5)
        
        # Row 4: Policy Network Stats (NEW)
        ttk.Label(learn_grid, text="Policy Network:", style='Stats.TLabel').grid(row=4, column=0, sticky=tk.W, padx=5)
        self.stat_policy = ttk.Label(learn_grid, text="Disabled", style='Stats.TLabel')
        self.stat_policy.grid(row=4, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(learn_grid, text="Carry Flips:", style='Stats.TLabel').grid(row=4, column=2, sticky=tk.W, padx=20)
        self.stat_carry_flips = ttk.Label(learn_grid, text="0", style='Stats.TLabel')
        self.stat_carry_flips.grid(row=4, column=3, sticky=tk.W, padx=5)
        
        # Row 5: Selection Methods (NEW)
        ttk.Label(learn_grid, text="Selection:", style='Stats.TLabel').grid(row=5, column=0, sticky=tk.W, padx=5)
        self.stat_selection = ttk.Label(learn_grid, text="NeuralNet:0 Random:0", style='Stats.TLabel')
        self.stat_selection.grid(row=5, column=1, columnspan=3, sticky=tk.W, padx=5)
        
        # History (last 10 attempts)
        history_frame = ttk.LabelFrame(stats_container, text="📜 Recent History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Treeview for history
        columns = ('restart', 'p', 'q', 'product', 'diff', 'time')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=8)
        
        self.history_tree.heading('restart', text='#')
        self.history_tree.heading('p', text='p')
        self.history_tree.heading('q', text='q')
        self.history_tree.heading('product', text='p × q')
        self.history_tree.heading('diff', text='Diff')
        self.history_tree.heading('time', text='Time')
        
        self.history_tree.column('restart', width=50)
        self.history_tree.column('p', width=80)
        self.history_tree.column('q', width=80)
        self.history_tree.column('product', width=100)
        self.history_tree.column('diff', width=60)
        self.history_tree.column('time', width=80)
        
        history_scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_elite_tab(self, parent):
        """Create the elite population tab."""
        colors = ModernStyle.current
        
        # Elite solutions list
        elite_container = ttk.Frame(parent, padding=10)
        elite_container.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(elite_container, text="Top solutions maintained for diversity:", 
                 style='Stats.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Treeview for elite
        columns = ('rank', 'p', 'q', 'product', 'diff', 'energy')
        self.elite_tree = ttk.Treeview(elite_container, columns=columns, show='headings', height=12)
        
        self.elite_tree.heading('rank', text='Rank')
        self.elite_tree.heading('p', text='p')
        self.elite_tree.heading('q', text='q')
        self.elite_tree.heading('product', text='p × q')
        self.elite_tree.heading('diff', text='Diff')
        self.elite_tree.heading('energy', text='Energy')
        
        self.elite_tree.column('rank', width=50)
        self.elite_tree.column('p', width=100)
        self.elite_tree.column('q', width=100)
        self.elite_tree.column('product', width=120)
        self.elite_tree.column('diff', width=80)
        self.elite_tree.column('energy', width=100)
        
        elite_scroll = ttk.Scrollbar(elite_container, orient=tk.VERTICAL, command=self.elite_tree.yview)
        self.elite_tree.configure(yscrollcommand=elite_scroll.set)
        
        self.elite_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        elite_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to copy selected
        self.elite_tree.bind('<Double-1>', self.copy_selected_elite)
        
        # Buttons
        btn_frame = ttk.Frame(elite_container)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="🔄 Refresh", command=self.refresh_elite).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📋 Copy Selected", command=self.copy_selected_elite).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📋 Copy Best", command=self.copy_best_solution).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📋 Copy All", command=self.copy_all_elites).pack(side=tk.LEFT, padx=2)
        
        # Hint label
        ttk.Label(btn_frame, text="(Double-click any row to copy)", 
                 style='Stats.TLabel').pack(side=tk.RIGHT, padx=10)
    
    def create_learning_tab(self, parent):
        """Create the learning visualization tab."""
        colors = ModernStyle.current
        
        learn_container = ttk.Frame(parent, padding=10)
        learn_container.pack(fill=tk.BOTH, expand=True)
        
        # Bit pattern statistics
        pattern_frame = ttk.LabelFrame(learn_container, text="📊 Bit Pattern Statistics", padding=10)
        pattern_frame.pack(fill=tk.X, pady=5)
        
        self.pattern_text = tk.Text(pattern_frame, height=8, font=('Consolas', 10),
                                    bg=colors['bg_secondary'], fg=colors['fg'])
        self.pattern_text.pack(fill=tk.X)
        
        # Top correlations
        corr_frame = ttk.LabelFrame(learn_container, text="🔗 Top Bit Correlations", padding=10)
        corr_frame.pack(fill=tk.X, pady=5)
        
        self.corr_text = tk.Text(corr_frame, height=6, font=('Consolas', 10),
                                 bg=colors['bg_secondary'], fg=colors['fg'])
        self.corr_text.pack(fill=tk.X)
        
        # Strategy usage
        strategy_frame = ttk.LabelFrame(learn_container, text="🎯 Initialization Strategies", padding=10)
        strategy_frame.pack(fill=tk.X, pady=5)
        
        self.strategy_text = tk.Text(strategy_frame, height=4, font=('Consolas', 10),
                                     bg=colors['bg_secondary'], fg=colors['fg'])
        self.strategy_text.pack(fill=tk.X)
        
        # Refresh button
        ttk.Button(learn_container, text="🔄 Refresh Learning Stats", 
                  command=self.refresh_learning).pack(pady=10)
    
    def create_policy_tab(self, parent):
        """Create the policy network tab."""
        colors = ModernStyle.current
        
        policy_container = ttk.Frame(parent, padding=10)
        policy_container.pack(fill=tk.BOTH, expand=True)
        
        # Policy Network Status
        status_frame = ttk.LabelFrame(policy_container, text="🤖 Policy Network Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)
        
        ttk.Label(status_grid, text="Status:", style='Stats.TLabel').grid(row=0, column=0, sticky=tk.W, padx=5)
        self.policy_net_status = ttk.Label(status_grid, text="Not Loaded", style='Stats.TLabel')
        self.policy_net_status.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(status_grid, text="Backend:", style='Stats.TLabel').grid(row=0, column=2, sticky=tk.W, padx=20)
        self.policy_backend = ttk.Label(status_grid, text="-", style='Stats.TLabel')
        self.policy_backend.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(status_grid, text="Bits:", style='Stats.TLabel').grid(row=1, column=0, sticky=tk.W, padx=5)
        self.policy_bits = ttk.Label(status_grid, text="-", style='Stats.TLabel')
        self.policy_bits.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(status_grid, text="Training Episodes:", style='Stats.TLabel').grid(row=1, column=2, sticky=tk.W, padx=20)
        self.policy_episodes = ttk.Label(status_grid, text="0", style='Stats.TLabel')
        self.policy_episodes.grid(row=1, column=3, sticky=tk.W, padx=5)
        
        # Selection Method Distribution
        selection_frame = ttk.LabelFrame(policy_container, text="🎯 Bit Selection Methods", padding=10)
        selection_frame.pack(fill=tk.X, pady=5)
        
        self.selection_text = tk.Text(selection_frame, height=4, font=('Consolas', 10),
                                      bg=colors['bg_secondary'], fg=colors['fg'])
        self.selection_text.pack(fill=tk.X)
        self.selection_text.insert(tk.END, "Selection method usage will appear here during solving...\n\n")
        self.selection_text.insert(tk.END, "• MLClauseLearner: 3-layer neural network makes ALL bit decisions\n")
        self.selection_text.insert(tk.END, "• NeuralNet: MLClauseLearner (3-layer, 512 hidden) makes ALL decisions\n")
        self.selection_text.insert(tk.END, "• Random: Fallback exploration (only if neural net unavailable)")
        self.selection_text.config(state=tk.DISABLED)
        
        # Carry Propagation Stats
        carry_frame = ttk.LabelFrame(policy_container, text="🔗 Carry-Aware Propagation", padding=10)
        carry_frame.pack(fill=tk.X, pady=5)
        
        self.carry_text = tk.Text(carry_frame, height=5, font=('Consolas', 10),
                                  bg=colors['bg_secondary'], fg=colors['fg'])
        self.carry_text.pack(fill=tk.X)
        self.carry_text.insert(tk.END, "Carry propagation helps single-bit flips cascade correctly.\n\n")
        self.carry_text.insert(tk.END, "When you flip bit i of p or q, the product changes.\n")
        self.carry_text.insert(tk.END, "Carry propagation finds additional bits to flip that\n")
        self.carry_text.insert(tk.END, "make the total change move toward N.")
        self.carry_text.config(state=tk.DISABLED)
        
        # Training Controls
        train_frame = ttk.LabelFrame(policy_container, text="🎓 Training Controls", padding=10)
        train_frame.pack(fill=tk.X, pady=5)
        
        train_row1 = ttk.Frame(train_frame)
        train_row1.pack(fill=tk.X, pady=2)
        
        ttk.Label(train_row1, text="Train on bit size:").pack(side=tk.LEFT)
        self.train_bits_var = tk.StringVar(value="16")
        ttk.Entry(train_row1, textvariable=self.train_bits_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(train_row1, text="Episodes:").pack(side=tk.LEFT, padx=(20, 0))
        self.train_episodes_entry = tk.StringVar(value="1000")
        ttk.Entry(train_row1, textvariable=self.train_episodes_entry, width=8).pack(side=tk.LEFT, padx=5)
        
        train_row2 = ttk.Frame(train_frame)
        train_row2.pack(fill=tk.X, pady=5)
        
        self.train_btn = ttk.Button(train_row2, text="🎓 Start Training", 
                                    command=self.train_policy_network_full, width=20)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_policy_btn = ttk.Button(train_row2, text="📂 Load Policy", 
                                          command=self.load_policy_file, width=15)
        self.load_policy_btn.pack(side=tk.LEFT, padx=5)
        
        # Training output
        self.train_output = scrolledtext.ScrolledText(train_frame, height=8, font=('Consolas', 9),
                                                       bg=colors['bg_secondary'], fg=colors['fg'])
        self.train_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
    
    def create_status_bar(self, parent):
        """Create the bottom status bar."""
        colors = ModernStyle.current
        
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.LEFT, padx=5)
        
        # Status message
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style='Stats.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Version
        ttk.Label(status_frame, text="v2.2", style='Stats.TLabel').pack(side=tk.RIGHT, padx=5)
    
    # === Event Handlers ===
    
    def toggle_theme(self):
        """Toggle between dark and light themes."""
        self.current_theme = 'light' if self.current_theme == 'dark' else 'dark'
        self.theme_btn.config(text="🌙 Dark" if self.current_theme == 'dark' else "☀️ Light")
        self.apply_theme()
        # Would need to rebuild UI for full theme change - simplified here
        messagebox.showinfo("Theme", f"Theme changed to {self.current_theme}. Restart app for full effect.")
    
    def show_help(self):
        """Show help dialog."""
        help_text = """
Quantum Annealing Factorization v2.2

This tool uses simulated quantum annealing with machine learning 
enhancements to factor integers.

Key Features:
• Adaptive temperature control (auto-reheats when stuck)
• Elite population for diversity
• Bit correlation learning
• Local search optimization
• Multiple initialization strategies
• Independent p/q encoding (p and q no longer forced to be complements)
• NEW: MLClauseLearner - 3-layer neural network (512 hidden) makes ALL bit decisions
• Neural network state saved/loaded from State File (persistent learning!)
• Legacy PolicyNetwork now optional - for initialization seeding only
• NEW: Carry-Aware Propagation - Flips cascade through multiplication

Policy Network:
• Train on small semiprimes to learn good bit flip strategies
• Uses curriculum learning (starts small, increases difficulty)
• Can use PyTorch (if available) or pure NumPy
• 40% policy-guided, 40% gradient-guided, 20% random exploration

Carry Propagation:
• When you flip bit i, the product p*q changes
• Carry propagation finds additional bits to flip
• Makes single decisions more impactful

Parameters:
• Triangle Pairs: Total pairs for BOTH factors (split: half for p, half for q)
  - Auto mode: Automatically calculates optimal pairs based on N
  - Manual: Uncheck "Auto" to set custom value
• Steps: Annealing steps per restart
• Reads: Samples per step
• Initial/Final Temp: Temperature schedule

Tips:
• Train the policy network before solving for better results
• Enable "Run until convergence" for automatic solving
• Check the Policy Net tab to see training progress
        """
        messagebox.showinfo("Help", help_text)
    
    def _calculate_optimal_pairs(self, n_int: int) -> tuple:
        """Calculate optimal number of triangle pairs for a given N.
        
        Returns: (total_pairs, bits_per_factor, info_string)
        """
        import math
        sqrt_n = int(math.isqrt(n_int)) + 1
        bits_per_factor = sqrt_n.bit_length() + 1  # +1 for safety margin
        total_pairs = 2 * bits_per_factor
        info = f"({bits_per_factor} for p, {bits_per_factor} for q)"
        return total_pairs, bits_per_factor, info
    
    def _on_n_changed(self, *args):
        """Called when N value changes - auto-update pairs if auto mode is on."""
        if not hasattr(self, 'auto_pairs_var') or not self.auto_pairs_var.get():
            return
        
        try:
            n_str = self.n_var.get().strip()
            if not n_str:
                return
            n_int = int(n_str)
            if n_int < 2:
                return
            
            total_pairs, bits_per_factor, info = self._calculate_optimal_pairs(n_int)
            self.pairs_var.set(str(total_pairs))
            self.pairs_info_label.config(text=info)
        except (ValueError, AttributeError):
            pass  # Invalid input, ignore
    
    def _on_auto_pairs_changed(self):
        """Called when auto checkbox is toggled."""
        if self.auto_pairs_var.get():
            # Auto mode ON - disable entry and recalculate
            self.pairs_entry.config(state='disabled')
            self._on_n_changed()  # Trigger recalculation
        else:
            # Auto mode OFF - enable entry for manual input
            self.pairs_entry.config(state='normal')
    
    def load_preset_2021(self):
        """Load preset for N=2021."""
        self.n_var.set("2021")
        # Let auto-calculation handle pairs if enabled
        if not self.auto_pairs_var.get():
            self.pairs_var.set("14")
        self.steps_var.set("80")
        self.reads_var.set("15")
        # Enable auto-scale temperature
        self.auto_temp_var.set(True)
        self._toggle_temp_entries()
        self.max_restarts_var.set("0")
        self.state_file_var.set("state_2021.json")
        total_pairs, bits, info = self._calculate_optimal_pairs(2021)
        self.log(f"📋 Loaded preset: N=2021 (factors: 43 × 47) - {total_pairs} pairs {info}", 'info')
    
    def load_preset_143(self):
        """Load preset for N=143."""
        self.n_var.set("143")
        # Let auto-calculation handle pairs if enabled
        if not self.auto_pairs_var.get():
            self.pairs_var.set("10")
        self.steps_var.set("50")
        self.reads_var.set("10")
        # Enable auto-scale temperature
        self.auto_temp_var.set(True)
        self._toggle_temp_entries()
        self.max_restarts_var.set("0")
        self.state_file_var.set("state_143.json")
        total_pairs, bits, info = self._calculate_optimal_pairs(143)
        self.log(f"📋 Loaded preset: N=143 (factors: 11 × 13) - {total_pairs} pairs {info}", 'info')
    
    def load_preset_15(self):
        """Load preset for N=15."""
        self.n_var.set("15")
        # Let auto-calculation handle pairs if enabled
        if not self.auto_pairs_var.get():
            self.pairs_var.set("6")
        self.steps_var.set("30")
        self.reads_var.set("5")
        # Enable auto-scale temperature
        self.auto_temp_var.set(True)
        self._toggle_temp_entries()
        self.max_restarts_var.set("0")
        self.state_file_var.set("state_15.json")
        total_pairs, bits, info = self._calculate_optimal_pairs(15)
        self.log(f"📋 Loaded preset: N=15 (factors: 3 × 5) - {total_pairs} pairs {info}", 'info')
    
    def load_custom(self):
        """Open dialog for custom N."""
        from tkinter import simpledialog
        n = simpledialog.askstring("Custom N", "Enter number to factor:")
        if n:
            try:
                n_int = int(n)
                self.n_var.set(n)
                # Use same auto-calculation as other presets
                total_pairs, bits, info = self._calculate_optimal_pairs(n_int)
                self.pairs_var.set(str(total_pairs))
                self.state_file_var.set(get_safe_state_filename(n_int))
                self.log(f"📋 Custom N={n} - {total_pairs} pairs {info}", 'info')
            except ValueError:
                messagebox.showerror("Error", "Invalid number")
    
    def _toggle_temp_entries(self):
        """Toggle temperature entry fields based on auto-scale checkbox."""
        if self.auto_temp_var.get():
            # Auto-scale enabled - disable manual entries
            self.init_temp_entry.config(state='disabled')
            self.final_temp_entry.config(state='disabled')
            self.init_temp_var.set("Auto")
            self.final_temp_var.set("Auto")
            self.temp_info_label.config(text="📐 Temp scales: 100×(1+log₂(N_bits))")
        else:
            # Manual mode - enable entries with default values
            self.init_temp_entry.config(state='normal')
            self.final_temp_entry.config(state='normal')
            self.init_temp_var.set("1000")
            self.final_temp_var.set("0.01")
            self.temp_info_label.config(text="⚠️ Manual mode - adjust for your N size")
    
    def _on_strategy_change(self, *args):
        """Callback when bit selection strategy sliders change."""
        transformer_pct = self.transformer_pct_var.get()
        hourglass_pct = self.hourglass_pct_var.get()
        random_pct = self.random_pct_var.get()
        
        # Update labels
        self.transformer_pct_label.config(text=f"{transformer_pct}%")
        self.hourglass_pct_label.config(text=f"{hourglass_pct}%")
        self.random_pct_label.config(text=f"{random_pct}%")
        
        # Update total
        total = transformer_pct + hourglass_pct + random_pct
        if total == 100:
            self.strategy_total_var.set(f"Total: {total}% ✓")
        else:
            self.strategy_total_var.set(f"Total: {total}% ⚠️")
    
    def _normalize_strategy(self):
        """Normalize strategy percentages to sum to 100%."""
        transformer_pct = self.transformer_pct_var.get()
        hourglass_pct = self.hourglass_pct_var.get()
        random_pct = self.random_pct_var.get()
        
        total = transformer_pct + hourglass_pct + random_pct
        
        if total == 0:
            # Default: even split
            self.transformer_pct_var.set(33)
            self.hourglass_pct_var.set(34)
            self.random_pct_var.set(33)
        else:
            # Normalize proportionally
            factor = 100.0 / total
            new_transformer = int(round(transformer_pct * factor))
            new_hourglass = int(round(hourglass_pct * factor))
            new_random = 100 - new_transformer - new_hourglass  # Ensure exact 100
            
            self.transformer_pct_var.set(new_transformer)
            self.hourglass_pct_var.set(new_hourglass)
            self.random_pct_var.set(new_random)
        
        # Update display
        self._on_strategy_change()
    
    def get_bit_selection_strategy(self) -> dict:
        """Get the current bit selection strategy percentages."""
        return {
            'transformer_pct': self.transformer_pct_var.get(),
            'hourglass_pct': self.hourglass_pct_var.get(),
            'random_pct': self.random_pct_var.get()
        }
    
    def _on_metro_change(self, *args):
        """Callback when Metropolis leniency slider changes."""
        value = self.metro_min_accept_var.get()
        self.metro_pct_label.config(text=f"{value*100:.0f}%")
    
    def _set_metro_leniency(self, value: float):
        """Set Metropolis leniency to a preset value."""
        self.metro_min_accept_var.set(value)
        self._on_metro_change()
    
    def get_metropolis_settings(self) -> dict:
        """Get the current Metropolis acceptance settings."""
        return {
            'min_accept_prob': self.metro_min_accept_var.get()
        }
    
    def browse_state_file(self):
        """Browse for state file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.state_file_var.set(filename)
    
    def browse_policy_file(self):
        """Browse for policy network file."""
        filename = filedialog.askopenfilename(
            defaultextension=".npz",
            filetypes=[("NumPy files", "*.npz"), ("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.policy_file_var.set(filename)
    
    def train_policy_network(self):
        """Quick train policy network with current settings."""
        episodes = int(self.train_episodes_var.get())
        self.train_policy_network_full(episodes)
    
    def train_policy_network_full(self, episodes=None):
        """Train policy network in background thread."""
        if episodes is None:
            episodes = int(self.train_episodes_entry.get())
        bits = int(self.train_bits_var.get()) if hasattr(self, 'train_bits_var') else 16
        
        self.train_btn.config(state=tk.DISABLED)
        self.policy_status_var.set("Training...")
        self.log(f"🎓 Starting policy network training: {episodes} episodes, {bits}-bit factors", 'info')
        
        def train_thread():
            try:
                from policy_network import PolicyNetworkTrainer, HAS_TORCH
                
                trainer = PolicyNetworkTrainer(n_bits=bits, hidden_dim=128)
                
                # Load existing if available
                policy_file = self.policy_file_var.get()
                if os.path.exists(policy_file):
                    try:
                        trainer.load(policy_file)
                        self.log(f"  Loaded existing policy from {policy_file}", 'info')
                    except:
                        pass
                
                # Update backend info
                self.root.after(0, lambda: self.policy_backend.config(
                    text="PyTorch" if HAS_TORCH else "NumPy"))
                self.root.after(0, lambda: self.policy_bits.config(text=str(bits)))
                
                # Train with progress updates
                stats = trainer.train(num_episodes=episodes, episode_length=100, print_interval=50)
                
                # Save
                trainer.save(policy_file)
                
                # Update UI
                def update_ui():
                    self.policy_status_var.set(f"Trained: {stats['final_difficulty']}-bit, {stats['total_successes']}/{stats['total_episodes']} solved")
                    self.policy_episodes.config(text=str(stats['total_episodes']))
                    self.policy_net_status.config(text=f"Trained ({stats['total_successes']/stats['total_episodes']:.0%} success)")
                    self.train_btn.config(state=tk.NORMAL)
                    self.log(f"✅ Training complete: {stats['total_successes']}/{stats['total_episodes']} solved", 'success')
                    
                    if hasattr(self, 'train_output'):
                        self.train_output.insert(tk.END, 
                            f"\n=== Training Complete ===\n"
                            f"Episodes: {stats['total_episodes']}\n"
                            f"Successes: {stats['total_successes']}\n"
                            f"Success Rate: {stats['total_successes']/stats['total_episodes']:.1%}\n"
                            f"Final Difficulty: {stats['final_difficulty']}-bit\n"
                            f"Saved to: {policy_file}\n")
                        self.train_output.see(tk.END)
                
                self.root.after(0, update_ui)
                
            except Exception as e:
                self.root.after(0, lambda: self.log(f"❌ Training error: {e}", 'error'))
                self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.policy_status_var.set("Error"))
                import traceback
                self.root.after(0, lambda: self.log(traceback.format_exc(), 'error'))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def load_policy_file(self):
        """Load a policy network file."""
        filename = filedialog.askopenfilename(
            defaultextension=".npz",
            filetypes=[("NumPy files", "*.npz"), ("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            try:
                from policy_network import PolicyNetwork, HAS_TORCH
                bits = int(self.train_bits_var.get()) if hasattr(self, 'train_bits_var') else 16
                policy = PolicyNetwork(n_bits=bits, hidden_dim=128)
                policy.load(filename)
                
                self.policy_file_var.set(filename)
                self.policy_status_var.set(f"Loaded: {filename}")
                self.policy_net_status.config(text="Loaded")
                self.policy_backend.config(text="PyTorch" if HAS_TORCH else "NumPy")
                self.policy_bits.config(text=str(bits))
                
                if hasattr(policy, 'episode_rewards'):
                    self.policy_episodes.config(text=str(len(policy.episode_rewards)))
                
                self.log(f"✅ Policy loaded from {filename}", 'success')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load policy: {e}")
    
    def log(self, message: str, tag: str = None):
        """Add message to output with optional formatting. Caps at 10,000 lines."""
        self.output_text.insert(tk.END, message + "\n", tag)
        
        # Cap log at 10,000 lines
        line_count = int(self.output_text.index('end-1c').split('.')[0])
        if line_count > 10000:
            # Delete oldest lines to bring it back to 10,000
            lines_to_delete = line_count - 10000
            self.output_text.delete('1.0', f'{lines_to_delete + 1}.0')
        
        if self.autoscroll_var.get():
            self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_output(self):
        """Clear the output text."""
        self.output_text.delete(1.0, tk.END)
    
    def save_log(self):
        """Save log to file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.output_text.get(1.0, tk.END))
            self.log(f"💾 Log saved to {filename}", 'success')
    
    def start_annealing(self):
        """Start the annealing process."""
        if self.is_running:
            return
        
        try:
            # Parse parameters
            N = int(self.n_var.get())
            num_pairs = int(self.pairs_var.get())
            num_steps = int(self.steps_var.get())
            num_reads = int(self.reads_var.get())
            
            # Temperature - None means auto-scale based on N
            if self.auto_temp_var.get():
                init_temp = None  # Auto-scale
                final_temp = None
            else:
                init_temp = float(self.init_temp_var.get())
                final_temp = float(self.final_temp_var.get())
            
            max_restarts = int(self.max_restarts_var.get())
            state_file = self.state_file_var.get()
            run_converge = self.converge_var.get()
            
            # Ensure state file name is safe (not too long)
            if len(state_file) > 200:
                state_file = get_safe_state_filename(N)
                self.state_file_var.set(state_file)
                self.log(f"⚠️ State filename was too long, using: {state_file}", 'warning')
            
            # Validate
            if N < 4:
                messagebox.showerror("Error", "N must be >= 4")
                return
            
            # Update UI
            self.is_running = True
            self.stop_flag = False
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.progress.start()
            self.status_var.set("Running...")
            self.start_time = time.time()
            
            # Reset statistics
            self.restart_count = 0
            self.best_diff = float('inf')
            self.history = []
            
            # Clear history tree
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            # Log start
            self.log("=" * 60, 'dim')
            self.log(f"🚀 Starting Quantum Annealing Factorization", 'info')
            self.log("=" * 60, 'dim')
            self.log(f"Target: N = {N}", 'info')
            half_pairs = num_pairs // 2
            self.log(f"Triangle pairs: {num_pairs} total ({half_pairs} for p, {half_pairs} for q)")
            self.log(f"  → p can be up to {2**half_pairs - 1}, q can be up to {2**half_pairs - 1}")
            self.log(f"Steps: {num_steps}, Reads: {num_reads}")
            if init_temp is None:
                n_bits = N.bit_length()
                auto_init = 100.0 * (1 + (n_bits.bit_length() if n_bits > 0 else 1))  # approx
                self.log(f"Temperature: Auto-scaled for {n_bits}-bit N (~{auto_init:.0f} → auto)")
            else:
                self.log(f"Temperature: {init_temp} → {final_temp}")
            self.log(f"Convergence mode: {run_converge}")
            self.log("")
            
            # Start thread
            self.annealing_thread = threading.Thread(
                target=self._run_annealing,
                args=(N, num_pairs, num_steps, num_reads, init_temp, final_temp, 
                      max_restarts if max_restarts > 0 else None, state_file, run_converge),
                daemon=True
            )
            self.annealing_thread.start()
            
            # Start update timer
            self.update_stats()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {e}")
    
    def _run_annealing(self, N, num_pairs, num_steps, num_reads, init_temp, final_temp, 
                       max_restarts, state_file, run_converge):
        """Run annealing in background thread."""
        import io
        
        try:
            # Import and create annealer
            from incremental_annealing_with_logging import IncrementalQuantumAnnealing
            
            log_file = "annealing_gui.log"
            # Pass state_file to init so it can load existing state before starting
            self.annealer = IncrementalQuantumAnnealing(
                N, num_pairs, log_file, 
                initial_temp=init_temp, 
                final_temp=final_temp,
                state_file=state_file
            )
            
            # Set bit selection strategy from GUI
            strategy = self.get_bit_selection_strategy()
            self.annealer.bit_selection_strategy = strategy
            self.log(f"🎛️ Bit selection: Transformer {strategy['transformer_pct']}%, "
                     f"Hourglass {strategy['hourglass_pct']}%, Random {strategy['random_pct']}%")
            
            # Set Metropolis acceptance settings from GUI
            metro_settings = self.get_metropolis_settings()
            self.annealer.metropolis_min_accept = metro_settings['min_accept_prob']
            self.log(f"🎰 Metropolis: Min accept = {metro_settings['min_accept_prob']*100:.0f}% "
                     f"({'lenient' if metro_settings['min_accept_prob'] > 0.1 else 'normal' if metro_settings['min_accept_prob'] > 0.02 else 'strict'})")
            
            # Load policy network if specified
            policy_file = self.policy_file_var.get() if hasattr(self, 'policy_file_var') else ""
            if policy_file and os.path.exists(policy_file):
                try:
                    from policy_network import PolicyNetwork, HAS_TORCH
                    n_bits = self.annealer.num_triangle_pairs  # bits for both p and q
                    policy = PolicyNetwork(n_bits=n_bits, hidden_dim=128)
                    policy.load(policy_file)
                    self.annealer.policy_network = policy
                    self.annealer.use_policy_network = True
                    self.log(f"✅ Policy network loaded: {policy_file}", 'success')
                except Exception as e:
                    self.log(f"⚠️ Could not load policy network: {e}", 'warning')
            
            # Capture stdout
            old_stdout = sys.stdout
            
            class OutputCapture:
                def __init__(self, gui):
                    self.gui = gui
                    self.buffer = ""
                
                def write(self, text):
                    self.buffer += text
                    if '\n' in text:
                        lines = self.buffer.split('\n')
                        for line in lines[:-1]:
                            if line.strip():
                                # Color code based on content
                                if '✓✓✓' in line or 'SUCCESS' in line or 'EXACT' in line:
                                    self.gui.log(line, 'success')
                                elif '🌟' in line or 'NEW BEST' in line:
                                    self.gui.log(line, 'warning')
                                elif 'Error' in line or 'ERROR' in line:
                                    self.gui.log(line, 'error')
                                elif '[' in line and ']' in line:
                                    self.gui.log(line, 'info')
                                else:
                                    self.gui.log(line)
                        self.buffer = lines[-1]
                
                def flush(self):
                    pass
            
            sys.stdout = OutputCapture(self)
            
            if run_converge:
                result = self.annealer.solve_until_convergence(
                    state_file=state_file,
                    num_steps=num_steps,
                    num_reads_per_step=num_reads,
                    checkpoint_interval=10,
                    max_restarts=max_restarts,
                    save_interval=5
                )
                if result:
                    config, energy, p, q = result
                    if p * q == N:
                        self.log(f"\n🎉 FACTORIZATION COMPLETE!", 'success')
                        self.log(f"   {N} = {p} × {q}", 'success')
            else:
                # Single run or multi-restart
                if max_restarts and max_restarts > 1:
                    config, energy = self.annealer.solve_with_restarts(
                        num_restarts=max_restarts,
                        num_steps=num_steps,
                        num_reads_per_step=num_reads,
                        checkpoint_interval=10
                    )
                else:
                    config, energy = self.annealer.incremental_solve(
                        num_steps=num_steps,
                        checkpoint_interval=10,
                        num_reads_per_step=num_reads
                    )
            
            sys.stdout = old_stdout
            
        except Exception as e:
            sys.stdout = sys.__stdout__
            self.log(f"❌ Error: {str(e)}", 'error')
            import traceback
            self.log(traceback.format_exc(), 'error')
        
        finally:
            # Update UI
            self.root.after(0, self._annealing_complete)
    
    def _annealing_complete(self):
        """Called when annealing completes."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()
        self.status_var.set("Complete")
        
        # Final refresh
        self.refresh_elite()
        self.refresh_learning()
    
    def stop_annealing(self):
        """Stop the annealing process."""
        self.stop_flag = True
        self.status_var.set("Stopping...")
        self.log("\n⏹️ Stop requested - saving progress...", 'warning')
        # The annealer will detect this via KeyboardInterrupt simulation
        # For now, we just set flag and let it finish current restart
    
    def resume_annealing(self):
        """Resume from saved state - always runs in convergence mode."""
        state_file = self.state_file_var.get()
        if os.path.exists(state_file):
            try:
                import json
                with open(state_file, 'r') as f:
                    saved_state = json.load(f)
                self.log(f"⏯️ Resuming from {state_file}", 'info')
                self.log(f"  Previous restarts: {saved_state.get('total_restarts', 0)}", 'info')
                self.log(f"  Learned clauses: {len(saved_state.get('learned_clauses', []))}", 'info')
                self.log(f"  Partial solutions: {len(saved_state.get('best_partial_solutions', []))}", 'info')
                self.log(f"  Elite population: {len(saved_state.get('elite_population', []))}", 'info')
                if saved_state.get('best_p') and saved_state.get('best_q'):
                    p, q = saved_state['best_p'], saved_state['best_q']
                    self.log(f"  Best so far: {p} × {q} = {p*q}", 'warning')
            except Exception as e:
                self.log(f"  (Could not read state preview: {e})", 'dim')
            
            # Force convergence mode ON when resuming
            self.converge_var.set(True)
            self.log(f"  [Auto] Convergence mode enabled for resume", 'info')
            
            self.start_annealing()
        else:
            messagebox.showinfo("Resume", f"No state file found: {state_file}")
    
    def update_stats(self):
        """Update statistics display periodically."""
        if not self.is_running:
            return
        
        # Update elapsed time
        if self.start_time:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            self.stat_elapsed.config(text=f"{mins}:{secs:02d}")
        
        # Update from annealer if available
        if self.annealer:
            # Restart count
            if hasattr(self.annealer, 'stuck_counter'):
                self.stat_stuck.config(text=str(self.annealer.stuck_counter))
            
            # Elite population
            if hasattr(self.annealer, 'elite_population'):
                self.stat_elite.config(text=str(len(self.annealer.elite_population)))
            
            # Learned clauses
            if hasattr(self.annealer, 'learned_clauses'):
                self.stat_clauses.config(text=str(len(self.annealer.learned_clauses)))
            
            # Correlations
            if hasattr(self.annealer, 'bit_correlations'):
                self.stat_correlations.config(text=str(len(self.annealer.bit_correlations)))
            
            # Temperature
            if hasattr(self.annealer, 'current_temp'):
                self.stat_temp.config(text=f"{self.annealer.current_temp:.1f}")
            
            # NEW: Tabu list size
            if hasattr(self.annealer, 'tabu_list'):
                self.stat_tabu.config(text=str(len(self.annealer.tabu_list)))
            
            # NEW: Nogood patterns
            if hasattr(self.annealer, 'nogood_patterns'):
                self.stat_nogoods.config(text=str(len(self.annealer.nogood_patterns)))
            
            # NEW: Bad bit combinations
            if hasattr(self.annealer, 'bad_bit_combos'):
                self.stat_bad_combos.config(text=str(len(self.annealer.bad_bit_combos)))
            
            # NEW: Policy Network status (legacy - for init only)
            if hasattr(self.annealer, 'policy_network') and self.annealer.policy_network is not None:
                self.stat_policy.config(text="Init ✓")
            else:
                self.stat_policy.config(text="Not used")
            
            # === MLClauseLearner Neural Network Status (PRIMARY) ===
            if hasattr(self.annealer, 'ml_clause_learner'):
                nn = self.annealer.ml_clause_learner
                self.nn_status_var.set(f"Status: Active ✓ (3-layer, 512 hidden)")
                self.nn_samples_var.set(f"Training samples: {nn.num_samples:,}")
                self.nn_patterns_var.set(f"Best patterns: {len(nn.best_patterns):,} | Replay: {len(nn.replay_buffer):,}")
            else:
                self.nn_status_var.set("Status: Not initialized")
                self.nn_samples_var.set("Training samples: 0")
                self.nn_patterns_var.set("Best patterns: 0")
            
            # NEW: Carry flips count
            if hasattr(self.annealer, 'carry_flip_count'):
                self.stat_carry_flips.config(text=str(self.annealer.carry_flip_count))
            
            # Selection method counters - Neural Network is now exclusive
            # "gradient" selection actually uses MLClauseLearner neural network
            nn_sel = getattr(self.annealer, 'gradient_selections', 0)  # Neural network selections
            random_sel = getattr(self.annealer, 'random_selections', 0)
            total = nn_sel + random_sel
            nn_pct = (100 * nn_sel / total) if total > 0 else 0
            self.stat_selection.config(text=f"NeuralNet:{nn_sel} ({nn_pct:.0f}%) | Random:{random_sel}")
            
            # NEW: Thresholds (both very_bad and clause - adaptive)
            if hasattr(self.annealer, 'very_bad_threshold'):
                clause_thresh = getattr(self.annealer, 'clause_threshold', 0)
                best_diff = getattr(self.annealer, 'best_diff_seen', float('inf'))
                N = getattr(self.annealer, 'N', 1)
                
                # Show improvement as ratio to N if we've made progress
                if best_diff != float('inf') and N > 0:
                    # Use Decimal for huge number division to avoid overflow
                    from decimal import Decimal
                    try:
                        ratio = float(Decimal(int(best_diff)) / Decimal(N))
                        pct = ratio * 100
                        if pct < 0.01:
                            self.stat_threshold.config(text=f"off: {ratio:.2e}")
                        else:
                            self.stat_threshold.config(text=f"off: {pct:.2f}%")
                    except:
                        self.stat_threshold.config(text="off: calc error")
                else:
                    self.stat_threshold.config(text="off: ∞ (no data)")
        
        # Schedule next update
        self.root.after(500, self.update_stats)
    
    def refresh_elite(self):
        """Refresh the elite population display."""
        # Clear current
        for item in self.elite_tree.get_children():
            self.elite_tree.delete(item)
        
        if self.annealer and hasattr(self.annealer, 'elite_population'):
            for i, elite in enumerate(self.annealer.elite_population):
                # Safely format energy (may be too large for float)
                energy = elite.get('energy', 0)
                try:
                    # Check if integer is too large for float (max ~1.8e308)
                    if isinstance(energy, int) and abs(energy) > 10**308:
                        # For extremely large integers, use string representation with magnitude
                        energy_str = f"~10^{len(str(abs(energy)))-1}"
                    else:
                        energy_str = f"{float(energy):.2f}"
                except (OverflowError, ValueError):
                    # Fallback: convert to string directly without float conversion
                    energy_str = str(energy)[:15] + "..." if len(str(energy)) > 15 else str(energy)
                
                self.elite_tree.insert('', 'end', values=(
                    i + 1,
                    elite.get('p', '?'),
                    elite.get('q', '?'),
                    elite.get('p', 0) * elite.get('q', 0),
                    elite.get('diff', '?'),
                    energy_str
                ))
    
    def refresh_learning(self):
        """Refresh learning statistics display."""
        if not self.annealer:
            return
        
        # Pattern statistics
        self.pattern_text.delete(1.0, tk.END)
        if hasattr(self.annealer, 'good_bit_patterns'):
            self.pattern_text.insert(tk.END, "Good patterns (bit → {0: count, 1: count}):\n")
            for i in sorted(self.annealer.good_bit_patterns.keys())[:10]:
                good = self.annealer.good_bit_patterns.get(i, {})
                bad = self.annealer.bad_bit_patterns.get(i, {})
                self.pattern_text.insert(tk.END, f"  Bit {i}: good={good}, bad={bad}\n")
        
        # NEW: Show bad bit combos
        if hasattr(self.annealer, 'bad_bit_combos') and self.annealer.bad_bit_combos:
            self.pattern_text.insert(tk.END, "\n🚫 Bad bit combinations (avoided):\n")
            sorted_bad = sorted(self.annealer.bad_bit_combos.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
            for (i, vi, j, vj), count in sorted_bad:
                self.pattern_text.insert(tk.END, f"  bit[{i}]={vi} & bit[{j}]={vj}: {count}× bad\n")
        
        # Correlations
        self.corr_text.delete(1.0, tk.END)
        if hasattr(self.annealer, 'bit_correlations') and self.annealer.bit_correlations:
            sorted_corr = sorted(self.annealer.bit_correlations.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
            self.corr_text.insert(tk.END, "Top correlations (bits that appear together):\n")
            for (i, vi, j, vj), count in sorted_corr:
                self.corr_text.insert(tk.END, f"  bit[{i}]={vi} & bit[{j}]={vj}: {count} times\n")
        
        # Strategy stats
        self.strategy_text.delete(1.0, tk.END)
        self.strategy_text.insert(tk.END, "Strategies: best_partial, elite_crossover, genetic, correlation_guided, sqrt_biased\n\n")
        self.strategy_text.insert(tk.END, f"Elite population size: {len(getattr(self.annealer, 'elite_population', []))}\n")
        self.strategy_text.insert(tk.END, f"Best partial solutions: {len(getattr(self.annealer, 'best_partial_solutions', []))}\n")
        self.strategy_text.insert(tk.END, f"Learned clauses: {len(getattr(self.annealer, 'learned_clauses', []))}\n")
        
        # NEW: Tabu/Nogood learning stats
        self.strategy_text.insert(tk.END, f"\n🔒 Tabu/Nogood Learning:\n")
        self.strategy_text.insert(tk.END, f"  Tabu list (bad configs): {len(getattr(self.annealer, 'tabu_list', []))}\n")
        self.strategy_text.insert(tk.END, f"  Nogood patterns: {len(getattr(self.annealer, 'nogood_patterns', []))}\n")
        self.strategy_text.insert(tk.END, f"  Bad bit combos: {len(getattr(self.annealer, 'bad_bit_combos', {}))}\n")
        
        # Helper to format big numbers (handles huge ints without float overflow)
        def fmt(n):
            if n == float('inf') or n == 'N/A':
                return "∞"
            if isinstance(n, int) and n > 10**15:
                # Use string manipulation for huge ints to avoid float overflow
                s = str(n)
                exp = len(s) - 1
                mantissa_str = s[0] + "." + s[1:3]
                return f"{mantissa_str}e{exp}"
            if isinstance(n, (int, float)) and abs(n) > 1e9:
                try:
                    exp = len(str(int(abs(n)))) - 1
                    mantissa = n / (10 ** exp)
                    return f"{mantissa:.2f}e{exp}"
                except OverflowError:
                    return f"~10^{len(str(n))-1}"
            return str(int(n) if isinstance(n, float) else n)
        
        bad_thresh = getattr(self.annealer, 'very_bad_threshold', float('inf'))
        self.strategy_text.insert(tk.END, f"  Very bad threshold: {fmt(bad_thresh)}\n")
        
        # Adaptive clause threshold info
        clause_thresh = getattr(self.annealer, 'clause_threshold', float('inf'))
        best_diff = getattr(self.annealer, 'best_diff_seen', float('inf'))
        N = getattr(self.annealer, 'N', 1)
        
        self.strategy_text.insert(tk.END, f"\n📊 ADAPTIVE THRESHOLDS:\n")
        self.strategy_text.insert(tk.END, f"  Best diff found: {fmt(best_diff)}\n")
        self.strategy_text.insert(tk.END, f"  Clause threshold: {fmt(clause_thresh)}\n")
        self.strategy_text.insert(tk.END, f"  Very bad threshold: {fmt(bad_thresh)}\n")
        
        # Show progress as ratio (using Decimal for huge numbers)
        if best_diff != float('inf') and N > 0:
            from decimal import Decimal
            try:
                ratio = float(Decimal(int(best_diff)) / Decimal(N))
                pct = ratio * 100
                self.strategy_text.insert(tk.END, f"\n📈 PROGRESS:\n")
                self.strategy_text.insert(tk.END, f"  Error: {pct:.4f}% of N\n")
                self.strategy_text.insert(tk.END, f"  (0% = exact factorization found)\n")
            except:
                self.strategy_text.insert(tk.END, f"\n📈 PROGRESS: calculation error\n")
        
        # NEW: Show nogood patterns if any
        if hasattr(self.annealer, 'nogood_patterns') and self.annealer.nogood_patterns:
            self.strategy_text.insert(tk.END, f"\n🚫 Learned Nogoods:\n")
            for pattern, diff in self.annealer.nogood_patterns[:3]:
                self.strategy_text.insert(tk.END, f"  Pattern: {pattern[:4]}... (diff={diff})\n")
        
        # NEW: Show learned clauses if any
        if hasattr(self.annealer, 'learned_clauses') and self.annealer.learned_clauses:
            self.strategy_text.insert(tk.END, f"\n📝 Learned Clauses (bad patterns):\n")
            for clause, energy, diff in self.annealer.learned_clauses[:3]:
                self.strategy_text.insert(tk.END, f"  {clause[:4]}... (diff={diff})\n")
    
    def copy_best_solution(self):
        """Copy best solution to clipboard."""
        if self.annealer and hasattr(self.annealer, 'elite_population') and self.annealer.elite_population:
            best = self.annealer.elite_population[0]
            text = f"{best['p']} × {best['q']} = {best['p'] * best['q']}"
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.log(f"📋 Copied best: {text[:100]}...", 'info')
    
    def copy_selected_elite(self, event=None):
        """Copy selected elite solution to clipboard."""
        selection = self.elite_tree.selection()
        if not selection:
            self.log("⚠️ No elite selected. Click on a row first.", 'warning')
            return
        
        # Get selected item
        item = selection[0]
        values = self.elite_tree.item(item, 'values')
        
        if values:
            rank = values[0]
            # Get the actual elite data from annealer
            if self.annealer and hasattr(self.annealer, 'elite_population'):
                idx = int(rank) - 1  # rank is 1-indexed
                if 0 <= idx < len(self.annealer.elite_population):
                    elite = self.annealer.elite_population[idx]
                    p = elite['p']
                    q = elite['q']
                    product = p * q
                    text = f"{p} × {q} = {product}"
                    self.root.clipboard_clear()
                    self.root.clipboard_append(text)
                    
                    # Show truncated version in log
                    p_str = str(p)[:30] + "..." if len(str(p)) > 30 else str(p)
                    q_str = str(q)[:30] + "..." if len(str(q)) > 30 else str(q)
                    self.log(f"📋 Copied elite #{rank}: {p_str} × {q_str}", 'info')
    
    def copy_all_elites(self):
        """Copy all elite solutions to clipboard."""
        if not self.annealer or not hasattr(self.annealer, 'elite_population'):
            self.log("⚠️ No elite population available.", 'warning')
            return
        
        if not self.annealer.elite_population:
            self.log("⚠️ Elite population is empty.", 'warning')
            return
        
        lines = ["=== ELITE SOLUTIONS ===\n"]
        for i, elite in enumerate(self.annealer.elite_population):
            p = elite['p']
            q = elite['q']
            product = p * q
            diff = elite.get('diff', 'N/A')
            lines.append(f"#{i+1}: {p} × {q} = {product}")
            lines.append(f"    diff = {diff}\n")
        
        text = "\n".join(lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.log(f"📋 Copied all {len(self.annealer.elite_population)} elite solutions to clipboard", 'info')


def main():
    """Main entry point."""
    root = tk.Tk()
    app = FactorizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
