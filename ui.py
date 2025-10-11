"""
ui.py - User Interface for Naval Combat Simulation
Complete military-style UI with all requested features
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

class StatusLogWindow(QDialog):
    """Status log window showing mission events"""
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Status Log")
        self.setGeometry(200, 200, 600, 400)
        self.setStyleSheet("QDialog { background-color: #0a0e1a; }")

        layout = QVBoxLayout()

        title = QLabel("MISSION STATUS LOG")
        title.setStyleSheet("""
            font-size: 16px; font-weight: bold; color: #00ff41; letter-spacing: 2px;
            padding: 10px; background-color: #1a1f2e; border: 1px solid #00ff41;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117; color: #00ff41;
                font-family: 'Courier New', monospace; font-size: 11px;
                border: 2px solid #00ff41; padding: 10px;
            }
        """)
        layout.addWidget(self.log_text)

        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 8px; background-color: #00ff41;
                color: #0a0e1a; border: none; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #00cc33; }
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

class CommsLinkWindow(QDialog):
    """Communications window showing nearby ships"""
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Communications Link")
        self.setGeometry(200, 200, 700, 500)
        self.setStyleSheet("QDialog { background-color: #0a0e1a; }")

        layout = QVBoxLayout()

        title = QLabel("NEARBY VESSELS")
        title.setStyleSheet("""
            font-size: 16px; font-weight: bold; color: #00ff41; letter-spacing: 2px;
            padding: 10px; background-color: #1a1f2e; border: 1px solid #00ff41;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Vessel Type", "Distance (m)", "Threat Level", "Speed (kts)", "Heading (°)"])
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0d1117; color: #00ff41;
                font-family: 'Courier New', monospace; font-size: 11px;
                border: 2px solid #00ff41; gridline-color: #00ff41;
            }
            QHeaderView::section {
                background-color: #1a1f2e; color: #00ff41; font-weight: bold;
                padding: 5px; border: 1px solid #00ff41;
            }
            QTableWidget::item { padding: 5px; }
            QTableWidget::item:selected { background-color: #2a3f3e; }
        """)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self.info_label = QLabel("Scan Range: 150m | Intercept Range: 100m")
        self.info_label.setStyleSheet("""
            font-size: 11px; color: #00ff41; padding: 5px;
            background-color: #1a1f2e; border: 1px solid #00ff41;
        """)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        btn_layout = QHBoxLayout()

        refresh_btn = QPushButton("⟳ REFRESH")
        refresh_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 8px; background-color: #1a4d1a;
                color: #00ff41; border: 2px solid #00ff41; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2a5d2a; }
        """)
        refresh_btn.clicked.connect(self.update_ships)
        btn_layout.addWidget(refresh_btn)

        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 8px; background-color: #00ff41;
                color: #0a0e1a; border: none; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #00cc33; }
        """)
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.update_ships()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ships)
        self.timer.start(2000)

    def update_ships(self):
        nearby = self.controller.get_nearby_ships()
        self.table.setRowCount(len(nearby))

        for i, ship in enumerate(nearby):
            self.table.setItem(i, 0, QTableWidgetItem(ship['vessel_type']))
            self.table.setItem(i, 1, QTableWidgetItem(f"{ship['distance']:.1f}"))

            threat_item = QTableWidgetItem(ship['threat_level'].upper())
            if ship['threat_level'] == 'confirmed':
                threat_item.setForeground(QColor(255, 0, 0))
            elif ship['threat_level'] == 'possible':
                threat_item.setForeground(QColor(255, 255, 255))
            elif ship['threat_level'] == 'neutral':
                threat_item.setForeground(QColor(0, 255, 65))
            else:
                threat_item.setForeground(QColor(128, 128, 128))
            self.table.setItem(i, 2, threat_item)

            self.table.setItem(i, 3, QTableWidgetItem(f"{ship['speed']:.1f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{ship['heading']:.0f}"))

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
                background-color: #0a0e1a; color: #00ff41;
                font-family: 'Courier New', monospace;
            }
        """)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #0a0e1a; }")

        container_widget = QWidget()
        layout = QVBoxLayout(container_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(40, 30, 40, 30)

        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: #1a1f2e; border: 2px solid #00ff41;
                border-radius: 10px; padding: 30px;
            }
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(15)

        title = QLabel("NAVAL COMBAT")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #00ff41; letter-spacing: 6px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(title)

        subtitle = QLabel("SIMULATION SYSTEM")
        subtitle.setStyleSheet("font-size: 18px; color: #00ff41; letter-spacing: 4px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(subtitle)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("background-color: #00ff41; max-height: 2px;")
        container_layout.addWidget(divider)

        version = QLabel("TACTICAL TRAINING INTERFACE v2.1 - AI ENHANCED")
        version.setStyleSheet("font-size: 11px; color: #00ff41; letter-spacing: 2px;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(version)

        container_layout.addSpacing(10)

        mission_label = QLabel("DIFFICULTY LEVEL")
        mission_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #00ff41; letter-spacing: 2px;")
        container_layout.addWidget(mission_label)

        self.mission_combo = QComboBox()
        self.mission_combo.addItems(["Patrol Boat", "Attack Vessel"])
        self.mission_combo.setStyleSheet("""
            QComboBox {
                font-size: 13px; padding: 8px; background-color: #2d4a2b;
                color: #00ff41; border: 2px solid #00ff41; border-radius: 5px;
                font-weight: bold; letter-spacing: 1px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow {
                image: none; border-left: 5px solid transparent;
                border-right: 5px solid transparent; border-top: 5px solid #00ff41;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d4a2b; color: #00ff41;
                selection-background-color: #3d5a3b; border: 2px solid #00ff41;
            }
        """)
        container_layout.addWidget(self.mission_combo)

        container_layout.addSpacing(10)

        self.start_btn = QPushButton("START SIMULATION")
        self.start_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px; padding: 12px; background-color: #00ff41;
                color: #0a0e1a; border: none; border-radius: 5px;
                font-weight: bold; letter-spacing: 2px;
            }
            QPushButton:hover { background-color: #00cc33; }
        """)
        container_layout.addWidget(self.start_btn)

        container_layout.addSpacing(10)

        features = [
            "• SINGLE PLAYER MODE",
            "• AI-ADAPTIVE SCENARIOS",
            "• MACHINE LEARNING ENABLED",
            "• TACTICAL DECISION MAKING",
            "• RANGE-BASED INTERCEPTION",
            "• COMMUNICATIONS LINK",
            "• STATUS LOG SYSTEM"
        ]

        for feature in features:
            feat_label = QLabel(feature)
            feat_label.setStyleSheet("font-size: 11px; color: #00ff41; letter-spacing: 1px;")
            feat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(feat_label)

        layout.addWidget(container)
        layout.addStretch()

        scroll.setWidget(container_widget)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)


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

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(50)

    def init_ui(self):
        self.setWindowTitle("Naval Combat Simulation - AI Enhanced")
        self.setGeometry(50, 50, 1400, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0e1a; }
            QWidget {
                background-color: #0a0e1a; color: #00ff41;
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
        top_bar.setStyleSheet("background-color: #0d1117; border-bottom: 1px solid #00ff41;")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(20, 10, 20, 10)

        status_container = QWidget()
        status_layout = QVBoxLayout(status_container)
        status_layout.setSpacing(2)
        status_layout.setContentsMargins(0, 0, 0, 0)

        status_title = QLabel("STATUS")
        status_title.setStyleSheet("font-size: 10px; color: #00ff41; letter-spacing: 2px;")
        status_layout.addWidget(status_title)

        self.enemy_label = QLabel("Enemies Detected: 0")
        self.enemy_label.setStyleSheet("font-size: 16px; color: #00ff41; font-weight: bold;")
        status_layout.addWidget(self.enemy_label)

        top_bar_layout.addWidget(status_container)
        top_bar_layout.addStretch()

        contacts_container = QWidget()
        contacts_layout = QVBoxLayout(contacts_container)
        contacts_layout.setSpacing(2)
        contacts_layout.setContentsMargins(0, 0, 0, 0)

        contacts_title = QLabel("CONTACTS")
        contacts_title.setStyleSheet("font-size: 10px; color: #00ff41; letter-spacing: 2px;")
        contacts_layout.addWidget(contacts_title)

        self.hostile_label = QLabel("0 HOSTILE")
        self.hostile_label.setStyleSheet("font-size: 16px; color: #ff0000; font-weight: bold;")
        contacts_layout.addWidget(self.hostile_label)

        top_bar_layout.addWidget(contacts_container)
        top_bar_layout.addStretch()

        active_indicator = QLabel("● ACTIVE")
        active_indicator.setStyleSheet("font-size: 14px; color: #00ff41; font-weight: bold;")
        top_bar_layout.addWidget(active_indicator)

        main_layout.addWidget(top_bar)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(0)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Radar container
        radar_container = QWidget()
        radar_container.setStyleSheet("background-color: #0a0e1a; border-right: 1px solid #00ff41;")
        radar_layout = QVBoxLayout(radar_container)
        radar_layout.setContentsMargins(10, 10, 10, 10)
        radar_layout.setSpacing(5)

        radar_header = QWidget()
        radar_header.setStyleSheet("background-color: #0d1117;")
        radar_header_layout = QHBoxLayout(radar_header)
        radar_header_layout.setContentsMargins(10, 5, 10, 5)

        sector_label = QLabel("SECTOR A-1")
        sector_label.setStyleSheet("font-size: 12px; color: #00ff41; font-weight: bold; letter-spacing: 2px;")
        radar_header_layout.addWidget(sector_label)
        radar_header_layout.addStretch()

        range_label = QLabel("RANGE: 500NM")
        range_label.setStyleSheet("font-size: 12px; color: #00ff41; letter-spacing: 2px;")
        radar_header_layout.addWidget(range_label)

        radar_layout.addWidget(radar_header)

        self.scene = QGraphicsScene(0, 0, 800, 600)
        self.scene.setBackgroundBrush(QBrush(QColor(10, 14, 26)))
        self.view = QGraphicsView(self.scene)
        self.view.setStyleSheet("border: 2px solid #00ff41; background-color: #0a0e1a;")
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.mousePressEvent = self.radar_click
        self.view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        radar_layout.addWidget(self.view)

        radar_footer = QWidget()
        radar_footer.setStyleSheet("background-color: #0d1117;")
        radar_footer_layout = QHBoxLayout(radar_footer)
        radar_footer_layout.setContentsMargins(10, 5, 10, 5)

        mode_label = QLabel("MODE: SURFACE")
        mode_label.setStyleSheet("font-size: 11px; color: #00ff41; letter-spacing: 2px;")
        radar_footer_layout.addWidget(mode_label)
        radar_footer_layout.addStretch()

        scale_label = QLabel("SCALE: 1:50000")
        scale_label.setStyleSheet("font-size: 11px; color: #00ff41; letter-spacing: 2px;")
        radar_footer_layout.addWidget(scale_label)

        radar_layout.addWidget(radar_footer)

        # Bottom command buttons
        button_container = QWidget()
        button_container.setStyleSheet("background-color: #0d1117; border-top: 1px solid #00ff41;")
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(5)
        button_layout.setContentsMargins(10, 8, 10, 8)

        # Comms Link button
        comms_btn = QPushButton("Comms Link")
        comms_btn.setStyleSheet("""
            QPushButton {
                font-size: 10px; padding: 6px 10px; background-color: #1a1f2e;
                color: #00ff41; border: 1px solid #00ff41; border-radius: 3px; letter-spacing: 1px;
            }
            QPushButton:hover { background-color: #2a3f3e; }
        """)
        comms_btn.clicked.connect(self.show_comms_link)
        button_layout.addWidget(comms_btn)

        # Status Log button
        status_btn = QPushButton("Status Log")
        status_btn.setStyleSheet("""
            QPushButton {
                font-size: 10px; padding: 6px 10px; background-color: #1a1f2e;
                color: #00ff41; border: 1px solid #00ff41; border-radius: 3px; letter-spacing: 1px;
            }
            QPushButton:hover { background-color: #2a3f3e; }
        """)
        status_btn.clicked.connect(self.show_status_log)
        button_layout.addWidget(status_btn)

        # Other buttons
        for name in ["Deploy Drone", "Request Support", "Change Course", "Status Report"]:
            btn = QPushButton(name)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 10px; padding: 6px 10px; background-color: #1a1f2e;
                    color: #00ff41; border: 1px solid #00ff41; border-radius: 3px; letter-spacing: 1px;
                }
                QPushButton:hover { background-color: #2a3f3e; }
            """)
            button_layout.addWidget(btn)

        radar_layout.addWidget(button_container)
        content_layout.addWidget(radar_container, stretch=3)

        # Control panel
        control_panel = QWidget()
        control_panel.setStyleSheet("background-color: #0d1117;")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(15, 15, 15, 15)

        cmd_header = QLabel("COMMAND CONTROLS")
        cmd_header.setStyleSheet("""
            font-size: 13px; font-weight: bold; color: #00ff41;
            letter-spacing: 2px; padding: 8px;
        """)
        control_layout.addWidget(cmd_header)

        ops_label = QLabel("TACTICAL OPERATIONS")
        ops_label.setStyleSheet("font-size: 10px; color: #00ff41; letter-spacing: 1px;")
        control_layout.addWidget(ops_label)

        self.pause_btn = QPushButton("▶ START")
        self.pause_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; padding: 12px; background-color: #00ff41;
                color: #0a0e1a; border: none; border-radius: 3px;
                font-weight: bold; letter-spacing: 2px;
            }
            QPushButton:hover { background-color: #00cc33; }
        """)
        self.pause_btn.clicked.connect(self.toggle_pause)
        control_layout.addWidget(self.pause_btn)

        self.status_label = QLabel("Status: Ready\nUse WASD to navigate to patrol zone")
        self.status_label.setStyleSheet("""
            font-size: 11px; padding: 10px; background-color: #1a1f2e;
            color: #00ff41; border: 1px solid #00ff41; border-radius: 3px;
        """)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(70)
        control_layout.addWidget(self.status_label)

        # Scrollable vessel details
        details_frame = QFrame()
        details_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1f2e; border: 1px solid #00ff41;
                border-radius: 3px; padding: 10px;
            }
        """)
        details_frame.setMaximumHeight(150)
        details_layout = QVBoxLayout(details_frame)

        details_title = QLabel("VESSEL DETAILS")
        details_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #00ff41; letter-spacing: 1px;")
        details_layout.addWidget(details_title)

        details_scroll = QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setStyleSheet("""
            QScrollArea {
                border: none; background-color: #1a1f2e;
            }
            QScrollBar:vertical {
                background-color: #0d1117; width: 10px;
            }
            QScrollBar::handle:vertical {
                background-color: #00ff41; border-radius: 5px;
            }
        """)

        self.details_label = QLabel("No vessel selected")
        self.details_label.setStyleSheet("font-size: 11px; color: #00ff41;")
        self.details_label.setWordWrap(True)
        details_scroll.setWidget(self.details_label)

        details_layout.addWidget(details_scroll)
        control_layout.addWidget(details_frame)

        # Action buttons
        action_frame = QFrame()
        action_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1f2e; border: 1px solid #00ff41;
                border-radius: 3px; padding: 10px;
            }
        """)
        action_layout = QVBoxLayout(action_frame)

        action_title = QLabel("ACTIONS")
        action_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #00ff41; letter-spacing: 1px;")
        action_layout.addWidget(action_title)

        self.intercept_btn = QPushButton("⊕ INTERCEPT VESSEL")
        self.intercept_btn.setStyleSheet("""
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
        self.intercept_btn.setEnabled(False)
        self.intercept_btn.clicked.connect(self.intercept_vessel)
        action_layout.addWidget(self.intercept_btn)

        self.ignore_btn = QPushButton("✓ MARK AS SAFE")
        self.ignore_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 8px; background-color: #1a4d1a;
                color: #00ff41; border: 2px solid #00ff41; border-radius: 3px;
                font-weight: bold; letter-spacing: 1px;
            }
            QPushButton:hover { background-color: #2a5d2a; }
            QPushButton:disabled {
                background-color: #2a2a2a; color: #555555; border-color: #333333;
            }
        """)
        self.ignore_btn.setEnabled(False)
        self.ignore_btn.clicked.connect(self.mark_safe)
        action_layout.addWidget(self.ignore_btn)

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

        control_layout.addWidget(action_frame)

        # Tactical options
        tactical_frame = QFrame()
        tactical_frame.setStyleSheet("border-top: 1px solid #00ff41; padding-top: 8px; margin-top: 5px;")
        tactical_layout = QVBoxLayout(tactical_frame)
        tactical_layout.setSpacing(3)

        tactical_options = ["• ENGAGE TARGETS", "• MONITOR RADAR", "• TACTICAL AWARENESS", "• AI ADAPTIVE MODE"]

        for option in tactical_options:
            opt_label = QLabel(option)
            opt_label.setStyleSheet("font-size: 10px; color: #00ff41; letter-spacing: 1px;")
            tactical_layout.addWidget(opt_label)

        control_layout.addWidget(tactical_frame)

        # Graph
        graph_frame = QFrame()
        graph_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1f2e; border: 1px solid #00ff41; border-radius: 3px;
            }
        """)
        graph_layout = QVBoxLayout(graph_frame)
        graph_layout.setContentsMargins(5, 5, 5, 5)

        graph_title = QLabel("THREAT ANALYSIS")
        graph_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #00ff41; letter-spacing: 1px; padding: 3px;")
        graph_layout.addWidget(graph_title)

        series = QLineSeries()
        for i in range(10):
            series.append(i, random.randint(0, 100))

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("")
        chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        chart.legend().hide()
        chart.setBackgroundBrush(QBrush(QColor(26, 31, 46)))

        axis_x = QValueAxis()
        axis_x.setRange(0, 10)
        axis_x.setTitleText("Time")
        axis_x.setLabelsColor(QColor(0, 255, 65))
        axis_x.setGridLineColor(QColor(0, 255, 65, 50))

        axis_y = QValueAxis()
        axis_y.setRange(0, 100)
        axis_y.setTitleText("Level")
        axis_y.setLabelsColor(QColor(0, 255, 65))
        axis_y.setGridLineColor(QColor(0, 255, 65, 50))

        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        chart_view.setMinimumHeight(150)

        graph_layout.addWidget(chart_view)
        control_layout.addWidget(graph_frame)

        # Mode and status
        mode_status_frame = QFrame()
        mode_status_frame.setStyleSheet("border-top: 1px solid #00ff41; padding-top: 8px; margin-top: 5px;")
        mode_status_layout = QVBoxLayout(mode_status_frame)
        mode_status_layout.setSpacing(3)

        mode_row = QHBoxLayout()
        mode_lbl = QLabel("MODE:")
        mode_lbl.setStyleSheet("font-size: 11px; color: #00ff41;")
        mode_row.addWidget(mode_lbl)
        mode_row.addStretch()

        mode_val = QLabel("COMBAT")
        mode_val.setStyleSheet("font-size: 11px; color: #00ff41; font-weight: bold;")
        mode_row.addWidget(mode_val)
        mode_status_layout.addLayout(mode_row)

        status_row = QHBoxLayout()
        status_lbl = QLabel("STATUS:")
        status_lbl.setStyleSheet("font-size: 11px; color: #00ff41;")
        status_row.addWidget(status_lbl)
        status_row.addStretch()

        self.status_val = QLabel("READY")
        self.status_val.setStyleSheet("font-size: 11px; color: #00ff41; font-weight: bold;")
        status_row.addWidget(self.status_val)
        mode_status_layout.addLayout(status_row)

        control_layout.addWidget(mode_status_frame)

        content_layout.addWidget(control_panel, stretch=1)
        main_layout.addLayout(content_layout)

    def show_comms_link(self):
        """Show communications link window"""
        dialog = CommsLinkWindow(self.controller, self)
        dialog.exec()

    def show_status_log(self):
        """Show status log window"""
        dialog = StatusLogWindow(self.controller, self)
        dialog.exec()

    def keyPressEvent(self, event):
        if self.controller.game_over or self.controller.paused:
            return

        if event.key() == Qt.Key.Key_W:
            self.controller.move_player('w')
        elif event.key() == Qt.Key.Key_S:
            self.controller.move_player('s')
        elif event.key() == Qt.Key.Key_A:
            self.controller.move_player('a')
        elif event.key() == Qt.Key.Key_D:
            self.controller.move_player('d')
        elif event.key() == Qt.Key.Key_Space:
            self.controller.move_player('space')

    def update_display(self):
        for item_list in self.graphics_items.values():
            for item in item_list:
                self.scene.removeItem(item)
        self.graphics_items.clear()

        pen = QPen(QColor(0, 255, 65, 30))
        for i in range(0, 801, 100):
            self.scene.addLine(i, 0, i, 600, pen)
        for i in range(0, 601, 100):
            self.scene.addLine(0, i, 800, i, pen)

        zr = self.controller.zone_rect
        zone_item = QGraphicsRectItem(zr["x"], zr["y"], zr["width"], zr["height"])
        zone_item.setPen(QPen(QColor(0, 255, 65), 2))
        zone_item.setBrush(QBrush(QColor(0, 255, 65, 10)))
        self.scene.addItem(zone_item)

        for idx, unit in enumerate(self.controller.units):
            if not unit.active:
                continue

            items = []

            if hasattr(unit, 'vessel_type') and unit.vessel_type == "Naval Patrol Vessel":
                triangle = QPolygonF([
                    QPointF(0, -15),
                    QPointF(-10, 10),
                    QPointF(10, 10)
                ])
                item = QGraphicsPolygonItem(triangle)

                if self.controller.in_patrol_zone:
                    item.setBrush(QBrush(QColor(0, 255, 65)))
                    item.setPen(QPen(QColor(0, 255, 65), 3))
                else:
                    item.setBrush(QBrush(QColor(100, 100, 100)))
                    item.setPen(QPen(QColor(80, 80, 80), 3))

                circle = QGraphicsEllipseItem(-25, -25, 50, 50)
                circle.setPen(QPen(QColor(0, 255, 65, 100), 1))
                circle.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                circle.setPos(unit.x, unit.y)
                self.scene.addItem(circle)
                items.append(circle)

                item.setPos(unit.x, unit.y)
            else:
                # Show gray for unscanned vessels
                if not unit.scanned:
                    color = QColor(128, 128, 128)
                    border = QColor(100, 100, 100)
                elif unit.threat_level == "neutral":
                    color = QColor(0, 255, 65)
                    border = QColor(0, 200, 50)
                elif unit.threat_level == "possible":
                    color = QColor(255, 255, 255)
                    border = QColor(200, 200, 200)
                else:  # confirmed
                    color = QColor(255, 0, 0)
                    border = QColor(200, 0, 0)

                item = QGraphicsEllipseItem(-10, -10, 20, 20)
                item.setBrush(QBrush(color))
                item.setPen(QPen(border, 2))
                item.setPos(unit.x, unit.y)

                ring = QGraphicsEllipseItem(-15, -15, 30, 30)
                ring.setPen(QPen(color, 1))
                ring.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                ring.setPos(unit.x, unit.y)
                self.scene.addItem(ring)
                items.append(ring)

                if self.controller.selected_unit == unit:
                    highlight = QGraphicsEllipseItem(-20, -20, 40, 40)
                    highlight.setPen(QPen(QColor(255, 255, 0), 3))
                    highlight.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                    highlight.setPos(unit.x, unit.y)
                    self.scene.addItem(highlight)
                    items.append(highlight)

            self.scene.addItem(item)
            items.append(item)
            self.graphics_items[idx] = items

        status = self.controller.get_status_info()
        self.enemy_label.setText(f"Enemies Detected: {status['total_threats']}")
        self.hostile_label.setText(f"{status['confirmed_threats']} HOSTILE")

    def radar_click(self, event):
        if not self.controller.in_patrol_zone:
            return

        scene_pos = self.view.mapToScene(event.pos())
        unit = self.controller.select_unit(scene_pos.x(), scene_pos.y())

        if unit:
            threat_text = {
                "neutral": "Non-Threatening",
                "possible": "Possible Threat",
                "confirmed": "Confirmed Threat",
                "unknown": "Unknown - Not Scanned"
            }

            distance = self.controller.get_distance(self.controller.player_ship, unit)
            unit.distance_from_base = distance

            in_intercept_range = distance <= self.controller.INTERCEPT_RANGE

            details = f"""Type: {unit.vessel_type}
Threat Level: {threat_text.get(unit.threat_level, 'Unknown')}
Speed: {unit.speed:.1f} knots
Heading: {unit.heading:.0f}°
Distance: {distance:.0f} m
Intercept Range: {'YES' if in_intercept_range else 'NO - Too Far'}
Scan Status: {'Scanned' if unit.scanned else 'Not Scanned'}"""

            self.details_label.setText(details)

            if self.controller.in_patrol_zone and unit.scanned:
                self.intercept_btn.setEnabled(in_intercept_range)
                self.ignore_btn.setEnabled(True)
                self.mark_threat_btn.setEnabled(True)
            else:
                self.intercept_btn.setEnabled(False)
                self.ignore_btn.setEnabled(False)
                self.mark_threat_btn.setEnabled(False)
        else:
            self.details_label.setText("No vessel selected")
            self.intercept_btn.setEnabled(False)
            self.ignore_btn.setEnabled(False)
            self.mark_threat_btn.setEnabled(False)

        self.update_display()

    def intercept_vessel(self):
        is_correct, threat_level, message = self.controller.intercept_vessel()

        if message == "Success":
            if is_correct:
                self.status_label.setText("Vessel intercepted!\nThreat neutralized.")
            else:
                self.status_label.setText("Vessel intercepted!\nWARNING: May not have been hostile.")
        else:
            self.status_label.setText(f"Interception failed:\n{message}")

        self.details_label.setText("No vessel selected")
        self.intercept_btn.setEnabled(False)
        self.ignore_btn.setEnabled(False)
        self.mark_threat_btn.setEnabled(False)
        self.update_display()

    def mark_safe(self):
        is_correct, threat_level, message = self.controller.mark_safe()

        if threat_level:
            if is_correct:
                self.status_label.setText("Vessel marked as safe\nCorrect assessment.")
            else:
                self.status_label.setText("Vessel marked as safe\nWARNING: Risk taken!")

        self.details_label.setText("No vessel selected")
        self.intercept_btn.setEnabled(False)
        self.ignore_btn.setEnabled(False)
        self.mark_threat_btn.setEnabled(False)
        self.update_display()

    def mark_as_threat(self):
        """Mark vessel as threat"""
        is_correct, threat_level, message = self.controller.mark_threat()

        if threat_level:
            if is_correct:
                self.status_label.setText("Vessel marked as threat\nCorrect assessment.")
            else:
                self.status_label.setText("Vessel marked as threat\nWARNING: May be friendly!")

        self.details_label.setText("No vessel selected")
        self.intercept_btn.setEnabled(False)
        self.ignore_btn.setEnabled(False)
        self.mark_threat_btn.setEnabled(False)
        self.update_display()

    def update_simulation(self):
        collision_occurred = self.controller.update_simulation()

        status = self.controller.get_status_info()

        if self.controller.in_patrol_zone:
            self.status_label.setText(
                f"Status: In Patrol Zone\nConfirmed Threats: {status['confirmed_threats']}\n"
                f"Accuracy: {status['accuracy']:.1%}\nClick vessels to interact"
            )
            self.status_val.setText("IN ZONE")
        else:
            self.status_label.setText(
                "Status: Outside Patrol Zone\nUse WASD to navigate\nPress SPACE to stop"
            )
            self.status_val.setText("OUT OF ZONE")

        self.update_display()

        if self.controller.game_over:
            self.intercept_btn.setEnabled(False)
            self.ignore_btn.setEnabled(False)
            self.mark_threat_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)

    def toggle_pause(self):
        paused = self.controller.toggle_pause()

        if paused:
            self.pause_btn.setText("▶ START")
            self.status_val.setText("PAUSED")
        else:
            self.pause_btn.setText("⏸ PAUSE")
            self.status_val.setText("ACTIVE")
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
        self.setFixedSize(500, 700)

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
