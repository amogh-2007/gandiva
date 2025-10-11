"""
ui.py - User Interface for Naval Combat Simulation
Complete military-style UI with all requested features.
Refined UI flow with patrol phase and streamlined controls.
"""

import sys
import random
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                             QComboBox, QLabel, QVBoxLayout, QHBoxLayout,
                             QGraphicsScene, QGraphicsView, QGraphicsEllipseItem,
                             QGraphicsPolygonItem, QFrame, QGraphicsRectItem,
                             QTextEdit, QScrollArea, QDialog, QTableWidget,
                             QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import QTimer, Qt, QPointF, QRectF, QSize
from PyQt6.QtGui import QBrush, QColor, QPen, QPolygonF, QPainter, QFont
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

from backend import SimulationController

# =============================================================================
# POPUP WINDOWS
# =============================================================================

class HailVesselDialog(QDialog):
    """Popup for hailing a vessel and seeing its response."""
    def __init__(self, vessel_type, hail_message, response_message, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Hailing: {vessel_type}")
        self.setGeometry(400, 400, 500, 250)
        self.setStyleSheet("QDialog { background-color: #0a192f; }")

        layout = QVBoxLayout()
        layout.setSpacing(15)

        title = QLabel("COMMUNICATION CHANNEL OPEN")
        title.setStyleSheet("""
            font-size: 14px; font-weight: bold; color: #64ffda; letter-spacing: 2px;
            padding: 10px; background-color: #112240; border: 1px solid #64ffda;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        hail_label = QLabel(f"OUTGOING HAIL:\n> {hail_message}")
        hail_label.setWordWrap(True)
        hail_label.setStyleSheet("font-size: 11px; color: #8892b0;")
        layout.addWidget(hail_label)
        
        response_label = QLabel(f"INCOMING RESPONSE:\n> {response_message}")
        response_label.setWordWrap(True)
        response_label.setStyleSheet("font-size: 12px; color: #ccd6f6; font-weight: bold;")
        layout.addWidget(response_label)
        
        layout.addStretch()

        close_btn = QPushButton("CLOSE CHANNEL")
        close_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 8px; background-color: #64ffda;
                color: #0a192f; border: none; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #57d8c0; }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)

class StatusLogWindow(QDialog):
    """Status log window showing mission events"""
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Status Log")
        self.setGeometry(200, 200, 600, 400)
        self.setStyleSheet("QDialog { background-color: #0a192f; }")

        layout = QVBoxLayout()

        title = QLabel("MISSION STATUS LOG")
        title.setStyleSheet("""
            font-size: 16px; font-weight: bold; color: #64ffda; letter-spacing: 2px;
            padding: 10px; background-color: #112240; border: 1px solid #64ffda;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0a192f; color: #64ffda;
                font-family: 'Courier New', monospace; font-size: 11px;
                border: 2px solid #64ffda; padding: 10px;
            }
        """)
        layout.addWidget(self.log_text)

        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 8px; background-color: #64ffda;
                color: #0a192f; border: none; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #57d8c0; }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)
        self.update_log()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_log)
        self.timer.start(1000)

    def update_log(self):
        log_text = "\n".join(self.controller.status_log)
        self.log_text.setText(log_text)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

class StatusReportWindow(QDialog):
    """Status report window showing placeholder backend data."""
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Status Report")
        self.setGeometry(300, 300, 500, 400)
        self.setStyleSheet("QDialog { background-color: #0a192f; }")

        layout = QVBoxLayout()

        title = QLabel("SYSTEM STATUS REPORT")
        title.setStyleSheet("""
            font-size: 16px; font-weight: bold; color: #64ffda; letter-spacing: 2px;
            padding: 10px; background-color: #112240; border: 1px solid #64ffda;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet("""
            QTextEdit {
                background-color: #0a192f; color: #64ffda;
                font-family: 'Courier New', monospace; font-size: 11px;
                border: 2px solid #64ffda; padding: 10px;
            }
        """)
        
        placeholder_text = (
            "Fetching data from backend...\n\n"
            "== TACTICAL OVERVIEW ==\n"
            "Mission Time: 00:00:00\n"
            "Overall Threat Level: STANDBY\n"
            "Player Accuracy: N/A\n"
            "AI Status: ADAPTIVE MODE ACTIVE\n\n"
            "== PLAYER VESSEL STATUS ==\n"
            "Hull Integrity: 100%\n"
            "Weapon Systems: ONLINE\n"
            "Engine Status: NOMINAL\n\n"
            "== ENVIRONMENT ==\n"
            "Sea State: 2 (Calm)\n"
            "Visibility: 10 NM\n"
            "Active Contacts: 0"
        )
        self.report_text.setText(placeholder_text)
        layout.addWidget(self.report_text)

        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 8px; background-color: #64ffda;
                color: #0a192f; border: none; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #57d8c0; }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)

# =============================================================================
# START MENU
# =============================================================================

class StartMenu(QWidget):
    """Start menu with military-style design"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #0a192f; color: #64ffda;
                font-family: 'Courier New', monospace;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.addStretch(1)

        container_frame = QFrame()
        container_frame.setStyleSheet("""
            QFrame {
                background-color: #112240; border: 2px solid #64ffda;
                border-radius: 10px; padding: 30px;
                max-width: 450px;
            }
        """)
        container_layout = QVBoxLayout(container_frame)
        container_layout.setSpacing(15)

        title = QLabel("NAVAL COMBAT")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #64ffda; letter-spacing: 6px; border: none; padding: 0;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(title)

        subtitle = QLabel("SIMULATION SYSTEM")
        subtitle.setStyleSheet("font-size: 18px; color: #ccd6f6; letter-spacing: 4px; border: none; padding: 0;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(subtitle)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("background-color: #64ffda; max-height: 2px; border: none;")
        container_layout.addWidget(divider)
        
        container_layout.addSpacing(20)

        mission_label = QLabel("DIFFICULTY LEVEL")
        mission_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #64ffda; letter-spacing: 2px; border: none; padding: 0;")
        container_layout.addWidget(mission_label)

        self.mission_combo = QComboBox()
        self.mission_combo.addItems(["Patrol Boat", "Attack Vessel"])
        self.mission_combo.setStyleSheet("""
            QComboBox {
                font-size: 13px; padding: 8px; background-color: #1d3b53;
                color: #64ffda; border: 2px solid #64ffda; border-radius: 5px;
                font-weight: bold; letter-spacing: 1px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow {
                image: none; border-left: 5px solid transparent;
                border-right: 5px solid transparent; border-top: 5px solid #64ffda;
                width: 0; height: 0; margin-right: 15px;
            }
            QComboBox QAbstractItemView {
                background-color: #1d3b53; color: #64ffda;
                selection-background-color: #2c4a63; border: 2px solid #64ffda;
            }
        """)
        container_layout.addWidget(self.mission_combo)

        container_layout.addSpacing(20)

        self.start_btn = QPushButton("START SIMULATION")
        self.start_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px; padding: 12px; background-color: #64ffda;
                color: #0a192f; border: none; border-radius: 5px;
                font-weight: bold; letter-spacing: 2px;
            }
            QPushButton:hover { background-color: #57d8c0; }
        """)
        container_layout.addWidget(self.start_btn)
        
        container_layout.addSpacing(10)

        # Center the container frame horizontally
        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        h_layout.addWidget(container_frame)
        h_layout.addStretch(1)
        
        main_layout.addLayout(h_layout)
        main_layout.addStretch(1)

# =============================================================================
# SIMULATION WINDOW
# =============================================================================

class SimulationWindow(QMainWindow):
    """Main simulation window with military-style radar display"""
    def __init__(self, mission_type="Patrol Boat"):
        super().__init__()
        self.mission_type = mission_type
        
        player_data = {"accuracy": 0.5, "reaction_time": 0.5}
        self.controller = SimulationController(mission_type, "novice", player_data)
        
        self.graphics_items = {}
        # A UI-only flag to track if we've done the one-time UI change after patrol phase.
        self.patrol_phase_ui_updated = False

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(50)

    def init_ui(self):
        self.setWindowTitle("Naval Combat Simulation - AI Enhanced")
        self.setGeometry(50, 50, 1400, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #0a192f; }
            QWidget {
                background-color: #0a192f; color: #ccd6f6;
                font-family: 'Courier New', monospace;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Top status bar
        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: #112240; border-bottom: 1px solid #64ffda;")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(20, 10, 20, 10)

        status_container = QWidget()
        status_layout = QVBoxLayout(status_container)
        status_layout.setSpacing(2)
        status_layout.setContentsMargins(0, 0, 0, 0)

        status_title = QLabel("STATUS")
        status_title.setStyleSheet("font-size: 10px; color: #64ffda; letter-spacing: 2px;")
        status_layout.addWidget(status_title)

        self.enemy_label = QLabel("Enemies Detected: 0")
        self.enemy_label.setStyleSheet("font-size: 16px; color: #ccd6f6; font-weight: bold;")
        status_layout.addWidget(self.enemy_label)

        top_bar_layout.addWidget(status_container)
        top_bar_layout.addStretch()

        contacts_container = QWidget()
        contacts_layout = QVBoxLayout(contacts_container)
        contacts_layout.setSpacing(2)
        contacts_layout.setContentsMargins(0, 0, 0, 0)

        contacts_title = QLabel("CONTACTS")
        contacts_title.setStyleSheet("font-size: 10px; color: #64ffda; letter-spacing: 2px;")
        contacts_layout.addWidget(contacts_title)

        self.hostile_label = QLabel("0 HOSTILE")
        self.hostile_label.setStyleSheet("font-size: 16px; color: #ff4646; font-weight: bold;")
        contacts_layout.addWidget(self.hostile_label)

        top_bar_layout.addWidget(contacts_container)
        top_bar_layout.addStretch()

        active_indicator = QLabel("● ACTIVE")
        active_indicator.setStyleSheet("font-size: 14px; color: #64ffda; font-weight: bold;")
        top_bar_layout.addWidget(active_indicator)

        main_layout.addWidget(top_bar)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(0)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Radar container
        radar_container = QWidget()
        radar_container.setStyleSheet("background-color: #0a192f; border-right: 1px solid #64ffda;")
        radar_layout = QVBoxLayout(radar_container)
        radar_layout.setContentsMargins(10, 10, 10, 10)
        radar_layout.setSpacing(5)
        
        self.scene = QGraphicsScene(0, 0, 800, 600)
        self.scene.setBackgroundBrush(QBrush(QColor(10, 25, 47)))
        self.view = QGraphicsView(self.scene)
        self.view.setStyleSheet("border: 2px solid #64ffda; background-color: #0a192f;")
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.mousePressEvent = self.radar_click
        self.view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.view.keyPressEvent = self.keyPressEvent
        self.view.keyReleaseEvent = self.keyReleaseEvent
        radar_layout.addWidget(self.view)

        # Bottom command buttons
        button_container = QWidget()
        button_container.setStyleSheet("background-color: #112240; border-top: 1px solid #64ffda;")
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(5)
        button_layout.setContentsMargins(10, 8, 10, 8)

        status_report_btn = QPushButton("Status Report")
        status_report_btn.setStyleSheet("""
            QPushButton {
                font-size: 10px; padding: 6px 10px; background-color: #112240;
                color: #64ffda; border: 1px solid #64ffda; border-radius: 3px; letter-spacing: 1px;
            }
            QPushButton:hover { background-color: #1d3b53; }
        """)
        status_report_btn.clicked.connect(self.show_status_report_window)
        button_layout.addWidget(status_report_btn)
        
        button_layout.addStretch()

        radar_layout.addWidget(button_container)
        content_layout.addWidget(radar_container, stretch=3)

        # Control panel
        control_panel = QWidget()
        control_panel.setStyleSheet("background-color: #112240;")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(15, 15, 15, 15)

        self.pause_btn = QPushButton("▶ RESUME")
        self.pause_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; padding: 12px; background-color: #64ffda;
                color: #0a192f; border: none; border-radius: 3px;
                font-weight: bold; letter-spacing: 2px;
            }
            QPushButton:hover { background-color: #57d8c0; }
        """)
        self.pause_btn.clicked.connect(self.toggle_pause)
        control_layout.addWidget(self.pause_btn)

        self.status_label = QLabel("Status: Ready\nUse WASD to navigate to patrol zone")
        self.status_label.setStyleSheet("""
            font-size: 11px; padding: 10px; background-color: #1d3b53;
            color: #64ffda; border: 1px solid #64ffda; border-radius: 3px;
        """)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(70)
        control_layout.addWidget(self.status_label)

        # Communications Panel
        comms_frame = QFrame()
        comms_frame.setStyleSheet("""
            QFrame {
                background-color: #1d3b53; border: 1px solid #64ffda;
                border-radius: 3px; padding: 10px;
            }
        """)
        comms_layout = QVBoxLayout(comms_frame)
        comms_title = QLabel("COMMUNICATIONS")
        comms_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #64ffda; letter-spacing: 1px;")
        comms_layout.addWidget(comms_title)
        
        self.comms_table = QTableWidget()
        self.comms_table.setColumnCount(2)
        self.comms_table.setHorizontalHeaderLabels(["Vessel", "Dist."])
        self.comms_table.setStyleSheet("""
            QTableWidget {
                background-color: transparent; color: #ccd6f6;
                font-family: 'Courier New', monospace; font-size: 10px;
                border: none; gridline-color: #64ffda;
            }
            QHeaderView::section {
                background-color: #112240; color: #64ffda; font-weight: bold;
                padding: 4px; border: 1px solid #64ffda;
            }
            QTableWidget::item { padding: 4px; }
            QTableWidget::item:selected { background-color: #2c4a63; }
        """)
        self.comms_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comms_table.verticalHeader().hide()
        comms_layout.addWidget(self.comms_table)
        
        control_layout.addWidget(comms_frame)


        # Vessel Details
        details_frame = QFrame()
        details_frame.setStyleSheet("""
            QFrame {
                background-color: #1d3b53; border: 1px solid #64ffda;
                border-radius: 3px; padding: 10px;
            }
        """)
        details_frame.setMinimumHeight(150)
        details_layout = QVBoxLayout(details_frame)
        details_title = QLabel("VESSEL DETAILS")
        details_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #64ffda; letter-spacing: 1px;")
        details_layout.addWidget(details_title)
        
        details_scroll = QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        self.details_label = QLabel("No vessel selected")
        self.details_label.setStyleSheet("font-size: 11px; color: #ccd6f6; background-color: transparent;")
        self.details_label.setWordWrap(True)

        details_scroll.setWidget(self.details_label)
        details_layout.addWidget(details_scroll)

        control_layout.addWidget(details_frame)

        # Action buttons
        action_layout = QVBoxLayout()
        action_layout.setContentsMargins(0,0,0,0)
        
        self.intercept_btn = QPushButton("⊕ INTERCEPT VESSEL")
        self.intercept_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 8px; background-color: #4a2525;
                color: #ff4646; border: 2px solid #ff4646; border-radius: 3px;
                font-weight: bold; letter-spacing: 1px;
            }
            QPushButton:hover { background-color: #6b3535; }
            QPushButton:disabled {
                background-color: #2a2a2a; color: #555555; border-color: #333333;
            }
        """)
        self.intercept_btn.setEnabled(False)
        self.intercept_btn.clicked.connect(self.intercept_vessel)
        action_layout.addWidget(self.intercept_btn)

        self.mark_safe_btn = QPushButton("✓ MARK AS SAFE")
        self.mark_safe_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 8px; background-color: #1d3b53;
                color: #64ffda; border: 2px solid #64ffda; border-radius: 3px;
                font-weight: bold; letter-spacing: 1px;
            }
            QPushButton:hover { background-color: #2c4a63; }
            QPushButton:disabled {
                background-color: #2a2a2a; color: #555555; border-color: #333333;
            }
        """)
        self.mark_safe_btn.setEnabled(False)
        self.mark_safe_btn.clicked.connect(self.mark_safe)
        action_layout.addWidget(self.mark_safe_btn)

        self.mark_threat_btn = QPushButton("⚠ MARK AS THREAT")
        self.mark_threat_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 8px; background-color: #4d4d00;
                color: #ffff00; border: 2px solid #ffff00; border-radius: 3px;
                font-weight: bold; letter-spacing: 1px;
            }
            QPushButton:hover { background-color: #5d5d00; }
            QPushButton:disabled {
                background-color: #2a2a2a; color: #555555; border-color: #333333;
            }
        """)
        self.mark_threat_btn.setEnabled(False)
        self.mark_threat_btn.clicked.connect(self.mark_as_threat)
        action_layout.addWidget(self.mark_threat_btn)

        self.distress_btn = QPushButton("🚨 DISTRESS CALL")
        self.distress_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 8px; background-color: #8b0000;
                color: #ff4444; border: 2px solid #ff0000; border-radius: 3px;
                font-weight: bold; letter-spacing: 1px;
            }
            QPushButton:hover { background-color: #a00000; }
            QPushButton:disabled {
                background-color: #2a2a2a; color: #555555; border-color: #333333;
            }
        """)
        self.distress_btn.setEnabled(False)
        self.distress_btn.clicked.connect(self.distress_call)
        action_layout.addWidget(self.distress_btn)
        
        control_layout.addLayout(action_layout)
        
        control_layout.addStretch() # Pushes everything up

        content_layout.addWidget(control_panel, stretch=1)
        main_layout.addLayout(content_layout)
        
        self.view.setFocus() # Set initial focus to the radar view

    def show_status_report_window(self):
        """Show status report window"""
        dialog = StatusReportWindow(self.controller, self)
        dialog.exec()

    def show_status_log(self):
        """Show status log window"""
        dialog = StatusLogWindow(self.controller, self)
        dialog.exec()

    def show_hail_dialog(self, unit):
        """Handles the logic for hailing a vessel and returns if suspicious."""
        hail_message = "Unidentified vessel, this is Naval Patrol. Identify yourself immediately."
        is_suspicious = False
        
        threat = unit.true_threat_level
        vessel_type = unit.vessel_type

        if threat == "confirmed":
            is_suspicious = True
            responses = [f"This is the warship '{vessel_type}'. Stay clear or you will be fired upon!", "...", "[Radio Silence]", "[Static followed by weapon system charging sounds]"]
            response_message = random.choice(responses)
        elif threat == "possible":
            is_suspicious = True
            responses = [f"This is private vessel '{vessel_type}'. State your intentions.", "We are on a private charter. We do not need to identify.", "...Stand by... We are experiencing engine trouble."]
            response_message = random.choice(responses)
        else: # neutral
            is_suspicious = False
            responses = [f"This is the fishing vessel '{vessel_type}'. Just hauling in a catch, over.", f"Roger that, patrol. This is '{vessel_type}', all is well.", f"Hey there! This is the '{vessel_type}', just enjoying the day."]
            response_message = random.choice(responses)
        
        self.controller.add_log(f"Hailed {vessel_type}. Response: '{response_message}'")
        
        dialog = HailVesselDialog(vessel_type, hail_message, response_message, self)
        dialog.exec()
        
        return is_suspicious

    def keyPressEvent(self, event):
        if self.controller.game_over:
            return

        key_map = {
            Qt.Key.Key_W: 'w',
            Qt.Key.Key_S: 's',
            Qt.Key.Key_A: 'a',
            Qt.Key.Key_D: 'd',
            Qt.Key.Key_Up: 'w',
            Qt.Key.Key_Down: 's',
            Qt.Key.Key_Left: 'a',
            Qt.Key.Key_Right: 'd',
        }
        if not event.isAutoRepeat() and event.key() in key_map:
            self.controller.set_key_state(key_map[event.key()], True)
        elif event.key() == Qt.Key.Key_Space:
            self.controller.move_player('space')

    def keyReleaseEvent(self, event):
        if self.controller.game_over:
            return

        key_map = {
            Qt.Key.Key_W: 'w',
            Qt.Key.Key_S: 's',
            Qt.Key.Key_A: 'a',
            Qt.Key.Key_D: 'd',
            Qt.Key.Key_Up: 'w',
            Qt.Key.Key_Down: 's',
            Qt.Key.Key_Left: 'a',
            Qt.Key.Key_Right: 'd',
        }
        if not event.isAutoRepeat() and event.key() in key_map:
            self.controller.set_key_state(key_map[event.key()], False)

    def update_display(self):
        for item_list in self.graphics_items.values():
            for item in item_list:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
        self.graphics_items.clear()
        
        # Add grid lines
        grid_pen = QPen(QColor(20, 35, 60), 1, Qt.PenStyle.SolidLine)
        for x in range(0, 801, 50):
            self.scene.addLine(x, 0, x, 600, grid_pen)
        for y in range(0, 601, 50):
            self.scene.addLine(0, y, 800, y, grid_pen)

        # Draw patrol zone with conditional coloring
        zr = self.controller.zone_rect
        zone_item = QGraphicsRectItem(zr["x"], zr["y"], zr["width"], zr["height"])
        zone_item.setPen(QPen(QColor(255, 70, 70, 150), 2, Qt.PenStyle.DashLine))
        if self.controller.patrol_phase_active:
            zone_item.setBrush(QBrush(QColor(255, 70, 70, 20)))
        else:
            zone_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.scene.addItem(zone_item)
        self.graphics_items['zone'] = [zone_item]

        # Always draw player ship
        player_unit = self.controller.player_ship
        if player_unit:
            player_items = []
            triangle = QPolygonF([QPointF(0, -15), QPointF(-10, 10), QPointF(10, 10)])
            player_item = QGraphicsPolygonItem(triangle)
            player_item.setBrush(QBrush(QColor(100, 255, 218))) # Cyan
            player_item.setPen(QPen(QColor(200, 255, 230), 2))
            player_item.setPos(player_unit.x, player_unit.y)
            self.scene.addItem(player_item)
            player_items.append(player_item)
            self.graphics_items[0] = player_items
            
        # Conditionally draw other units if patrol phase is over
        if not self.controller.patrol_phase_active:
            for idx, unit in enumerate(self.controller.units):
                if not unit.active or unit == self.controller.player_ship:
                    continue
                
                items = []
                color = QColor(136, 146, 176) # Light slate for unknown
                border = QColor(100, 110, 140)

                if unit.scanned:
                    if unit.threat_level == "neutral":
                        color = QColor(100, 255, 218) # Cyan
                        border = QColor(150, 255, 230)
                    elif unit.threat_level == "possible":
                        color = QColor(255, 255, 255) # White
                        border = QColor(200, 200, 200)
                    else:  # confirmed
                        color = QColor(255, 70, 70) # Red
                        border = QColor(200, 50, 50)

                item = QGraphicsEllipseItem(-8, -8, 16, 16)
                item.setBrush(QBrush(color))
                item.setPen(QPen(border, 2))
                item.setPos(unit.x, unit.y)
                self.scene.addItem(item)
                items.append(item)

                if self.controller.selected_unit == unit:
                    highlight = QGraphicsEllipseItem(-15, -15, 30, 30)
                    highlight.setPen(QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine))
                    highlight.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                    highlight.setPos(unit.x, unit.y)
                    self.scene.addItem(highlight)
                    items.append(highlight)
                
                self.graphics_items[idx + 1] = items

        status = self.controller.get_status_info()
        self.enemy_label.setText(f"Enemies Detected: {status['total_threats']}")
        self.hostile_label.setText(f"{status['confirmed_threats']} HOSTILE")
        self.update_comms_panel()


    def radar_click(self, event):
        if self.controller.patrol_phase_active or not self.controller.in_patrol_zone:
            self.controller.selected_unit = None
            self.details_label.setText("No vessel selected")
            self.intercept_btn.setEnabled(False)
            self.mark_safe_btn.setEnabled(False)
            self.mark_threat_btn.setEnabled(False)
            self.distress_btn.setEnabled(False)
            return

        scene_pos = self.view.mapToScene(event.pos())
        unit = self.controller.select_unit(scene_pos.x(), scene_pos.y())

        if unit:
            distance = self.controller.get_distance(self.controller.player_ship, unit)
            in_intercept_range = distance <= self.controller.INTERCEPT_RANGE
            
            # Show hail dialog and get result
            is_suspicious = self.show_hail_dialog(unit)
            
            # Enable buttons based on hail result and range
            self.intercept_btn.setEnabled(in_intercept_range)
            self.mark_safe_btn.setEnabled(not is_suspicious)
            self.mark_threat_btn.setEnabled(is_suspicious)
            self.distress_btn.setEnabled(True)

            threat_text = unit.threat_level.capitalize() if unit.scanned else "Unknown"
            details = (f"Type: {unit.vessel_type}\n"
                       f"Threat Level: {threat_text}\n"
                       f"Distance: {distance:.0f} m\n\n"
                       f"Crew Size: {unit.crew_count}\n"
                       f"COMMUNICATION LOGGED.")
            self.details_label.setText(details)
        else:
            self.details_label.setText("No vessel selected")
            self.intercept_btn.setEnabled(False)
            self.mark_safe_btn.setEnabled(False)
            self.mark_threat_btn.setEnabled(False)
            self.distress_btn.setEnabled(False)
        self.update_display()

    def intercept_vessel(self):
        is_correct, threat_level, message = self.controller.intercept_vessel()
        self.details_label.setText("No vessel selected")
        self.intercept_btn.setEnabled(False)
        self.mark_safe_btn.setEnabled(False)
        self.mark_threat_btn.setEnabled(False)
        self.distress_btn.setEnabled(False)
        self.update_display()
        
    def distress_call(self):
        message = self.controller.distress_call()
        self.details_label.setText(message)
        self.intercept_btn.setEnabled(False)
        self.mark_safe_btn.setEnabled(False)
        self.mark_threat_btn.setEnabled(False)
        self.distress_btn.setEnabled(False)
        self.update_display()

    def mark_safe(self):
        is_correct, threat_level, message = self.controller.mark_safe()
        self.details_label.setText("No vessel selected")
        self.intercept_btn.setEnabled(False)
        self.mark_safe_btn.setEnabled(False)
        self.mark_threat_btn.setEnabled(False)
        self.distress_btn.setEnabled(False)
        self.update_display()

    def mark_as_threat(self):
        is_correct, threat_level, message = self.controller.mark_threat()
        self.details_label.setText("No vessel selected")
        self.intercept_btn.setEnabled(False)
        self.mark_safe_btn.setEnabled(False)
        self.mark_threat_btn.setEnabled(False)
        self.distress_btn.setEnabled(False)
        self.update_display()

    def update_simulation(self):
        if not self.controller.game_over:
            self.controller.update_simulation()
        
        status = self.controller.get_status_info()
        
        # Check if the patrol phase just ended to update UI elements
        if not self.patrol_phase_ui_updated and not self.controller.patrol_phase_active:
            self.patrol_phase_ui_updated = True
            self.pause_btn.setText("⏸ PAUSE")
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatioByExpanding)

        if self.controller.in_patrol_zone:
            self.status_label.setText(
                f"Status: In Patrol Zone\nConfirmed Threats: {status['confirmed_threats']}\n"
                f"Accuracy: {status['accuracy']:.1%}\nClick vessels to interact"
            )
        else:
            self.status_label.setText(
                "Status: Outside Patrol Zone\nUse WASD to navigate to the red zone."
            )

        self.update_display()
        if self.controller.game_over:
            self.timer.stop()
            
    def update_comms_panel(self):
        nearby = self.controller.get_nearby_ships()
        self.comms_table.setRowCount(len(nearby))
        for i, ship in enumerate(nearby):
            self.comms_table.setItem(i, 0, QTableWidgetItem(ship['vessel_type']))
            self.comms_table.setItem(i, 1, QTableWidgetItem(f"{ship['distance']:.0f}"))


    def toggle_pause(self):
        # Allow resume from initial pause, but not re-pausing during patrol phase
        if self.controller.patrol_phase_active:
            self.controller.paused = False # Start the simulation
            self.pause_btn.setText("⏸ PAUSE")
            self.view.setFocus()
            return
            
        paused = self.controller.toggle_pause()
        if paused:
            self.pause_btn.setText("▶ RESUME")
        else:
            self.pause_btn.setText("⏸ PAUSE")
            self.view.setFocus()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

class NavalSimApp(QMainWindow):
    """Main application"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Naval Combat Simulation")
        self.show_start_menu()

    def show_start_menu(self):
        self.start_menu = StartMenu()
        self.start_menu.start_btn.clicked.connect(self.start_simulation)
        self.setCentralWidget(self.start_menu)
        self.setMinimumSize(600, 750)
        self.resize(600, 750)

    def start_simulation(self):
        mission_type = self.start_menu.mission_combo.currentText()
        self.sim_window = SimulationWindow(mission_type)
        self.sim_window.show()
        self.close()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = NavalSimApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()