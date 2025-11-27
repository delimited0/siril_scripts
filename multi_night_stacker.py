#!/usr/bin/env python3
"""
Multi-Night Stacker GUI for Siril

A comprehensive GUI application for automated multi-night image stacking using Siril.
Automatically detects and processes multiple set folders (set1, set2, etc.) with 
individual calibration, then combines and stacks all nights together.

Features:
- Auto-detection of set1, set2, set3, etc. folders
- Synthetic bias calibration per set (no dark frames required)
- Flat field correction per set
- Combines all calibrated lights with symbolic links
- Global registration across all nights
- Configurable sigma rejection stacking
- OSC camera support with debayering

Requirements:
- sirilpy
- PyQt5

License: MIT
Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QTextEdit, QFileDialog, 
    QGroupBox, QGridLayout, QMessageBox, QProgressBar, QCheckBox,
    QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

try:
    import sirilpy
    from sirilpy import LogColor
    SIRILPY_AVAILABLE = True
except ImportError:
    SIRILPY_AVAILABLE = False


class SirilWorker(QThread):
    """Worker thread for running Siril commands."""
    
    log_message = pyqtSignal(str, str)  # message, color
    progress_update = pyqtSignal(int)  # percentage
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self.siril = None
    
    def run(self):
        """Execute the task function."""
        try:
            result = self.task_func(self, *self.args, **self.kwargs)
            if result is None or result:
                self.finished.emit(True, "Task completed successfully")
            else:
                self.finished.emit(False, "Task completed with errors")
        except Exception as e:
            self.log_message.emit(f"Error: {str(e)}", "red")
            self.finished.emit(False, str(e))
    
    def cmd(self, *args):
        """Execute Siril command with logging."""
        cmd_str = " ".join(str(arg) for arg in args)
        self.log_message.emit(f"$ {cmd_str}", "blue")
        try:
            if self.siril:
                self.siril.cmd(*args)
        except Exception as e:
            self.log_message.emit(f"Command failed: {e}", "red")
            raise


class MultiNightStackerGUI(QMainWindow):
    """Main application window."""
    
    def __init__(self, siril_instance=None):
        super().__init__()
        self.siril = siril_instance
        self.working_dir = None
        self.worker = None
        self.detected_sets = []
        
        self.init_ui()
        self.load_settings()
        
        # Auto-populate working directory from Siril
        if self.siril:
            try:
                wd = self.siril.get_siril_wd()
                if wd and Path(wd).exists():
                    wd_path = Path(wd)
                    
                    # Check if we're in a subdirectory
                    # If so, use the parent directory as working directory
                    if wd_path.name.startswith('set') or wd_path.name in ['lights', 'flats', 'process']:
                        wd_path = wd_path.parent
                        self.log(f"Detected subdirectory, using parent: {wd_path}", "orange")
                    
                    self.working_dir = str(wd_path)
                    self.dir_label.setText(str(wd_path))
                    self.detect_sets()
                    self.log(f"Working directory auto-detected: {wd_path}", "green")
            except AttributeError as e:
                self.log(f"Siril not connected properly: {e}", "red")
            except Exception as e:
                self.log(f"Could not auto-detect working directory: {e}", "orange")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Multi-Night Stacker for Siril")
        self.setGeometry(100, 100, 900, 750)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Directory selection
        dir_group = QGroupBox("Working Directory")
        dir_layout = QVBoxLayout()
        
        dir_select_layout = QHBoxLayout()
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setWordWrap(True)
        dir_button = QPushButton("Select Directory")
        dir_button.clicked.connect(self.select_directory)
        dir_select_layout.addWidget(self.dir_label, 1)
        dir_select_layout.addWidget(dir_button)
        dir_layout.addLayout(dir_select_layout)
        
        # Detected sets label
        self.sets_label = QLabel("No sets detected")
        self.sets_label.setStyleSheet("color: #666; font-style: italic;")
        dir_layout.addWidget(self.sets_label)
        
        dir_group.setLayout(dir_layout)
        main_layout.addWidget(dir_group)
        
        # Sequence settings
        seq_group = QGroupBox("Sequence Settings")
        seq_layout = QGridLayout()
        
        seq_layout.addWidget(QLabel("Sequence Name:"), 0, 0)
        self.seq_name_edit = QLineEdit()
        # Default to today's date in YYYYMMDD_seq format
        default_name = datetime.now().strftime("%Y%m%d_seq")
        self.seq_name_edit.setText(default_name)
        self.seq_name_edit.setToolTip("Name for combined sequence (e.g., 20251116_seq)")
        seq_layout.addWidget(self.seq_name_edit, 0, 1)
        
        seq_group.setLayout(seq_layout)
        main_layout.addWidget(seq_group)
        
        # Calibration settings
        calib_group = QGroupBox("Calibration Settings")
        calib_layout = QGridLayout()
        
        calib_layout.addWidget(QLabel("Synthetic Bias Coefficient:"), 0, 0)
        self.bias_coeff_spin = QDoubleSpinBox()
        self.bias_coeff_spin.setRange(0, 100)
        self.bias_coeff_spin.setValue(8)
        self.bias_coeff_spin.setDecimals(1)
        self.bias_coeff_spin.setToolTip("Multiplier for OFFSET value (e.g., 8 for Poseidon C Pro)")
        calib_layout.addWidget(self.bias_coeff_spin, 0, 1)
        
        self.use_flats_check = QCheckBox("Use Flat Frames")
        self.use_flats_check.setChecked(True)
        self.use_flats_check.setToolTip("Each set must have a flats/ folder")
        calib_layout.addWidget(self.use_flats_check, 1, 0, 1, 2)
        
        self.debayer_check = QCheckBox("Debayer (OSC Camera)")
        self.debayer_check.setChecked(True)
        self.debayer_check.setToolTip("Enable for color cameras (OSC/DSLR)")
        calib_layout.addWidget(self.debayer_check, 2, 0, 1, 2)
        
        calib_group.setLayout(calib_layout)
        main_layout.addWidget(calib_group)
        
        # Stacking settings
        stack_group = QGroupBox("Stacking Settings")
        stack_layout = QGridLayout()
        
        stack_layout.addWidget(QLabel("Sigma High (rejection):"), 0, 0)
        self.sigma_high_spin = QDoubleSpinBox()
        self.sigma_high_spin.setRange(0.1, 10.0)
        self.sigma_high_spin.setValue(3.0)
        self.sigma_high_spin.setDecimals(1)
        self.sigma_high_spin.setToolTip("High sigma threshold for rejection")
        stack_layout.addWidget(self.sigma_high_spin, 0, 1)
        
        stack_layout.addWidget(QLabel("Sigma Low (rejection):"), 1, 0)
        self.sigma_low_spin = QDoubleSpinBox()
        self.sigma_low_spin.setRange(0.1, 10.0)
        self.sigma_low_spin.setValue(3.0)
        self.sigma_low_spin.setDecimals(1)
        self.sigma_low_spin.setToolTip("Low sigma threshold for rejection")
        stack_layout.addWidget(self.sigma_low_spin, 1, 1)
        
        self.normalize_check = QCheckBox("Output Normalization")
        self.normalize_check.setChecked(True)
        self.normalize_check.setToolTip("Normalize output histogram")
        stack_layout.addWidget(self.normalize_check, 2, 0, 1, 2)
        
        self.rgb_equal_check = QCheckBox("RGB Equalization")
        self.rgb_equal_check.setChecked(True)
        self.rgb_equal_check.setToolTip("Equalize RGB channels (for color images)")
        stack_layout.addWidget(self.rgb_equal_check, 3, 0, 1, 2)
        
        stack_group.setLayout(stack_layout)
        main_layout.addWidget(stack_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setEnabled(False)
        self.start_button.setStyleSheet("font-weight: bold; padding: 8px;")
        button_layout.addWidget(self.start_button)
        
        save_preset_btn = QPushButton("Save Preset")
        save_preset_btn.clicked.connect(self.save_preset)
        button_layout.addWidget(save_preset_btn)
        
        load_preset_btn = QPushButton("Load Preset")
        load_preset_btn.clicked.connect(self.load_preset)
        button_layout.addWidget(load_preset_btn)
        
        main_layout.addLayout(button_layout)
        
        # Log output
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 9))
        self.log_text.setMinimumHeight(200)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        self.log("Multi-Night Stacker initialized", "green")
        
        if not SIRILPY_AVAILABLE:
            self.log("WARNING: sirilpy not available. Siril integration will not work.", "red")
    
    def log(self, message: str, color: str = "black"):
        """Add message to log with color."""
        color_map = {
            "black": "#000000",
            "red": "#FF0000",
            "green": "#00AA00",
            "blue": "#0000FF",
            "orange": "#FF8800",
        }
        
        hex_color = color_map.get(color, color_map["black"])
        self.log_text.append(f'<span style="color: {hex_color};">{message}</span>')
    
    def detect_sets(self):
        """Detect set1, set2, etc. folders in working directory."""
        if not self.working_dir:
            return
        
        working_path = Path(self.working_dir)
        self.detected_sets = []
        
        # Look for set1, set2, set3, etc.
        i = 1
        while True:
            set_path = working_path / f"set{i}"
            if not set_path.exists():
                break
            
            # Verify it has lights folder
            lights_path = set_path / "lights"
            if lights_path.exists():
                self.detected_sets.append(f"set{i}")
                self.log(f"Detected: {set_path.name}", "blue")
            else:
                self.log(f"Skipping {set_path.name}: no lights/ folder", "orange")
            
            i += 1
        
        if self.detected_sets:
            sets_text = f"Detected sets: {', '.join(self.detected_sets)} ({len(self.detected_sets)} total)"
            self.sets_label.setText(sets_text)
            self.sets_label.setStyleSheet("color: green; font-weight: bold;")
            self.start_button.setEnabled(True)
        else:
            self.sets_label.setText("No valid sets detected (expecting set1/, set2/, etc. with lights/ folders)")
            self.sets_label.setStyleSheet("color: red;")
            self.start_button.setEnabled(False)
    
    def select_directory(self):
        """Select working directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        
        if directory:
            self.working_dir = directory
            self.dir_label.setText(directory)
            
            # Detect sets
            self.detect_sets()
            
            if not self.detected_sets:
                QMessageBox.warning(
                    self, "No Sets Found",
                    "No valid set folders detected.\n\n"
                    "Expected structure:\n"
                    "  set1/\n"
                    "    lights/\n"
                    "    flats/\n"
                    "  set2/\n"
                    "    lights/\n"
                    "    flats/\n"
                    "  ..."
                )
            else:
                self.log(f"Working directory set: {directory}", "green")
    
    def start_processing(self):
        """Start the full processing workflow."""
        if not self.working_dir:
            QMessageBox.warning(self, "No Directory", "Please select a working directory first.")
            return
        
        if not self.detected_sets:
            QMessageBox.warning(self, "No Sets", "No valid sets detected. Please check directory structure.")
            return
        
        if not SIRILPY_AVAILABLE:
            QMessageBox.critical(self, "Missing Dependency", 
                                 "sirilpy is not available. This script must be run from Siril.")
            return
        
        seq_name = self.seq_name_edit.text().strip()
        if not seq_name:
            QMessageBox.warning(self, "No Sequence Name", "Please enter a sequence name.")
            return
        
        # Confirm with user
        reply = QMessageBox.question(
            self, "Start Processing",
            f"Process {len(self.detected_sets)} sets?\n\n"
            f"Sets: {', '.join(self.detected_sets)}\n"
            f"Sequence name: {seq_name}\n"
            f"Sigma rejection: {self.sigma_low_spin.value()}/{self.sigma_high_spin.value()}\n\n"
            f"This will:\n"
            f"1. Calibrate each set individually\n"
            f"2. Combine all calibrated lights\n"
            f"3. Register across all nights\n"
            f"4. Stack final result\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Start worker thread
        self.worker = SirilWorker(self.process_workflow)
        self.worker.log_message.connect(self.log)
        self.worker.progress_update.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()
    
    def process_workflow(self, worker: SirilWorker):
        """Main processing workflow executed in worker thread."""
        try:
            worker.log_message.emit("=== Starting Multi-Night Processing ===", "green")
            worker.siril = self.siril
            
            seq_name = self.seq_name_edit.text().strip()
            
            # Change to working directory
            worker.cmd("cd", self.working_dir)
            worker.progress_update.emit(5)
            
            # Create multi_night_combined directory
            combined_dir = Path(self.working_dir) / "multi_night_combined"
            combined_dir.mkdir(exist_ok=True)
            worker.log_message.emit(f"Created directory: {combined_dir}", "blue")
            
            # Process each set
            num_sets = len(self.detected_sets)
            base_progress = 5
            progress_per_set = 60 // num_sets if num_sets > 0 else 0
            
            for idx, set_name in enumerate(self.detected_sets):
                worker.log_message.emit(f"\n=== Processing {set_name} ===", "green")
                self.process_set(worker, set_name)
                worker.progress_update.emit(base_progress + (idx + 1) * progress_per_set)
            
            worker.log_message.emit("\n=== Combining All Nights ===", "green")
            self.combine_sequences(worker, seq_name)
            worker.progress_update.emit(70)
            
            worker.log_message.emit("\n=== Registering Across All Nights ===", "green")
            self.register_combined(worker, seq_name)
            worker.progress_update.emit(85)
            
            worker.log_message.emit("\n=== Stacking Final Result ===", "green")
            self.stack_combined(worker, seq_name)
            worker.progress_update.emit(100)
            
            worker.log_message.emit("\n=== Processing Complete! ===", "green")
            worker.log_message.emit(f"Final result: {self.working_dir}/{seq_name}_stacked.fit", "green")
            
            # Close Siril
            worker.cmd("close")
            
        except Exception as e:
            worker.log_message.emit(f"Error in workflow: {e}", "red")
            raise
    
    def process_set(self, worker: SirilWorker, set_name: str):
        """Process a single set folder."""
        set_path = Path(self.working_dir) / set_name
        
        # Process flats if enabled
        if self.use_flats_check.isChecked():
            flats_path = set_path / "flats"
            if flats_path.exists():
                worker.log_message.emit(f"Processing flats for {set_name}...", "blue")
                
                worker.cmd("cd", str(flats_path))
                worker.cmd("convert", "flat", "-out=../process")
                worker.cmd("cd", "../process")
                worker.cmd("calibrate", "flat")
                worker.cmd("stack", "pp_flat", "rej", "3", "3", "-norm=mul")
                worker.cmd("cd", self.working_dir)
            else:
                worker.log_message.emit(f"Warning: No flats folder in {set_name}", "orange")
        
        # Process lights
        worker.log_message.emit(f"Processing lights for {set_name}...", "blue")
        
        lights_path = set_path / "lights"
        worker.cmd("cd", str(lights_path))
        worker.cmd("convert", "light", "-out=../process")
        worker.cmd("cd", "../process")
        
        # Build calibration command
        bias_coeff = int(self.bias_coeff_spin.value())
        calib_args = ["calibrate", "light", f'-bias="={bias_coeff}*$OFFSET"']
        
        if self.use_flats_check.isChecked() and (set_path / "process" / "pp_flat_stacked.fit").exists():
            calib_args.append("-flat=pp_flat_stacked")
        
        if self.debayer_check.isChecked():
            calib_args.extend(["-cfa", "-equalize_cfa", "-debayer"])
        
        worker.cmd(*calib_args)
        worker.cmd("cd", self.working_dir)
        
        worker.log_message.emit(f"Completed {set_name}", "green")
    
    def combine_sequences(self, worker: SirilWorker, seq_name: str):
        """Combine all calibrated lights using symbolic links with renamed files."""
        combined_dir = Path(self.working_dir) / "multi_night_combined"
        
        frame_counter = 1
        
        # For each set, create symbolic links to pp_light files with new naming
        for set_name in self.detected_sets:
            set_process_dir = Path(self.working_dir) / set_name / "process"
            pp_light_files = sorted(set_process_dir.glob("pp_light_*.fit"))
            
            if pp_light_files:
                worker.log_message.emit(f"Linking {len(pp_light_files)} files from {set_name}", "blue")
                
                # Create symbolic links with unified naming
                for fit_file in pp_light_files:
                    link_name = combined_dir / f"{seq_name}_{frame_counter:05d}.fit"
                    if link_name.exists():
                        link_name.unlink()  # Remove existing link
                    link_name.symlink_to(fit_file)
                    frame_counter += 1
            else:
                worker.log_message.emit(f"Warning: No pp_light files found in {set_name}", "orange")
        
        worker.log_message.emit(f"Created {frame_counter - 1} symbolic links", "green")
    
    def register_combined(self, worker: SirilWorker, seq_name: str):
        """Register all frames across all nights."""
        combined_dir = Path(self.working_dir) / "multi_night_combined"
        worker.cmd("cd", str(combined_dir))
        
        worker.log_message.emit("Registering combined sequence...", "blue")
        worker.cmd("register", seq_name)
        
        worker.cmd("cd", self.working_dir)
    
    def stack_combined(self, worker: SirilWorker, seq_name: str):
        """Stack the registered combined sequence."""
        combined_dir = Path(self.working_dir) / "multi_night_combined"
        worker.cmd("cd", str(combined_dir))
        
        # Build stack command
        sigma_low = self.sigma_low_spin.value()
        sigma_high = self.sigma_high_spin.value()
        
        stack_args = ["stack", f"r_{seq_name}", "rej", str(sigma_low), str(sigma_high)]
        stack_args.append("-norm=addscale")
        
        if self.normalize_check.isChecked():
            stack_args.append("-output_norm")
        
        if self.rgb_equal_check.isChecked():
            stack_args.append("-rgb_equal")
        
        stack_args.extend(["-out=../" + seq_name + "_stacked"])
        
        worker.cmd(*stack_args)
        
        worker.cmd("cd", self.working_dir)
    
    def on_processing_finished(self, success: bool, message: str):
        """Handle processing completion."""
        self.start_button.setEnabled(True)
        
        if success:
            seq_name = self.seq_name_edit.text().strip()
            QMessageBox.information(
                self, "Processing Complete",
                f"Multi-night stacking complete!\n\n"
                f"Processed {len(self.detected_sets)} nights\n"
                f"Final result: {seq_name}_stacked.fit\n\n"
                f"You can now open the result in Siril for further processing."
            )
        else:
            QMessageBox.warning(self, "Processing Error", f"Processing failed: {message}")
    
    def save_preset(self):
        """Save current settings to preset file."""
        preset_data = {
            "sequence_name": self.seq_name_edit.text(),
            "bias_coefficient": self.bias_coeff_spin.value(),
            "use_flats": self.use_flats_check.isChecked(),
            "debayer": self.debayer_check.isChecked(),
            "sigma_high": self.sigma_high_spin.value(),
            "sigma_low": self.sigma_low_spin.value(),
            "output_normalization": self.normalize_check.isChecked(),
            "rgb_equalization": self.rgb_equal_check.isChecked(),
        }
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", "", "JSON Files (*.json)"
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(preset_data, f, indent=2)
            self.log(f"Preset saved: {file_path}", "green")
    
    def load_preset(self):
        """Load settings from preset file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Preset", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    preset_data = json.load(f)
                
                self.seq_name_edit.setText(preset_data.get("sequence_name", datetime.now().strftime("%Y%m%d_seq")))
                self.bias_coeff_spin.setValue(preset_data.get("bias_coefficient", 8))
                self.use_flats_check.setChecked(preset_data.get("use_flats", True))
                self.debayer_check.setChecked(preset_data.get("debayer", True))
                self.sigma_high_spin.setValue(preset_data.get("sigma_high", 3.0))
                self.sigma_low_spin.setValue(preset_data.get("sigma_low", 3.0))
                self.normalize_check.setChecked(preset_data.get("output_normalization", True))
                self.rgb_equal_check.setChecked(preset_data.get("rgb_equalization", True))
                
                self.log(f"Preset loaded: {file_path}", "green")
                
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load preset: {e}")
    
    def save_settings(self):
        """Save current settings to config file."""
        config_path = Path.home() / ".multi_night_stacker_config.json"
        
        settings = {
            "last_directory": self.working_dir,
            "bias_coefficient": self.bias_coeff_spin.value(),
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=2)
        except:
            pass
    
    def load_settings(self):
        """Load settings from config file."""
        config_path = Path.home() / ".multi_night_stacker_config.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                
                if "bias_coefficient" in settings:
                    self.bias_coeff_spin.setValue(settings["bias_coefficient"])
                    
            except:
                pass
    
    def closeEvent(self, event):
        """Handle window close."""
        self.save_settings()
        event.accept()


def main():
    """Main entry point."""
    # Ensure required dependencies are installed
    try:
        sirilpy.ensure_installed(
            "PyQt5",
            version_constraints=[None]
        )
    except Exception as e:
        print(f"Error ensuring dependencies: {e}")
        return 1
    
    # Create Siril interface and connect
    try:
        siril = sirilpy.SirilInterface()
        siril.connect()
    except Exception as e:
        print(f"Error connecting to Siril: {e}")
        print("Make sure this script is run from Siril's Scripts menu.")
        return 1
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = MultiNightStackerGUI(siril_instance=siril)
    window.show()
    
    result = app.exec_()
    return 0 if result == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
