"""
ui.py - User Interface for Naval Combat Simulation
Enhanced with bottom communication panel and distress call functionality
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                             QComboBox, QLabel, QVBoxLayout, QHBoxLayout,
                             QGraphicsScene, QGraphicsView, QGraphicsEllipseItem,
                             QGraphicsPolygonItem, QFrame, QGraphicsRectItem,
                             QTextEdit, QDialog, QTableWidget,
                             QTableWidgetItem, QHeaderView, QGridLayout, QSplitter)
from PyQt6.QtCore import QTimer, Qt, QPointF
from PyQt6.QtGui import QBrush, QColor, QPen, QPolygonF, QPainter, QLinearGradient, QGradient
from backend import SimulationController

# =============================================================================
# ENHANCED POPUP WINDOWS
# =============================================================================

class CommunicationWindow(QDialog):
    """Enhanced Communication window for vessel interactions"""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Communication Channel")
        self.setGeometry(300, 200, 800, 600)
        self.setStyleSheet("QDialog { background-color: #0a192f; }")

        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Title
        title = QLabel(" VESSEL COMMUNICATION SYSTEM ")
        title.setStyleSheet("""
            font-size: 16px; font-weight: bold; color: #64ffda; letter-spacing: 2px;
            padding: 10px; background-color: #112240; border: 1px solid #64ffda;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Nearby vessels label
        list_label = QLabel("NEARBY VESSELS:")
        list_label.setStyleSheet("font-size: 12px; color: #64ffda; font-weight: bold;")
        layout.addWidget(list_label)

        self.vessel_table = QTableWidget()
        self.vessel_table.setColumnCount(5)  # Added ID column
        self.vessel_table.setHorizontalHeaderLabels(["ID", "Vessel Type", "Distance", "Threat", "Status"])
        self.vessel_table.setStyleSheet("""
            QTableWidget {
                background-color: #112240; color: #ccd6f6;
                font-family: 'Courier New', monospace; font-size: 11px;
                border: 2px solid #64ffda; gridline-color: #64ffda;
            }
            QHeaderView::section {
                background-color: #1d3b53; color: #64ffda; font-weight: bold;
                padding: 6px; border: 1px solid #64ffda;
            }
            QTableWidget::item { padding: 6px; }
            QTableWidget::item:selected { background-color: #2c4a63; }
        """)
        self.vessel_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.vessel_table.verticalHeader().hide()
        self.vessel_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.vessel_table.itemSelectionChanged.connect(self.on_vessel_selected)
        layout.addWidget(self.vessel_table)

        # Message display area with AI scenario integration
        msg_label = QLabel("COMMUNICATION LOG & AI ANALYSIS:")
        msg_label.setStyleSheet("font-size: 12px; color: #64ffda; font-weight: bold;")
        layout.addWidget(msg_label)

        self.message_display = QTextEdit()
        self.message_display.setReadOnly(True)
        self.message_display.setStyleSheet("""
            QTextEdit {
                background-color: #0a192f; color: #64ffda;
                font-family: 'Courier New', monospace; font-size: 11px;
                border: 2px solid #64ffda; padding: 10px;
            }
        """)
        self.message_display.setMaximumHeight(200)
        layout.addWidget(self.message_display)

        # Enhanced action buttons
        button_layout = QHBoxLayout()

        hail_btn = QPushButton(" HAIL VESSEL")
        hail_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 8px; background-color: #1d3b53;
                color: #64ffda; border: 2px solid #64ffda; border-radius: 3px;
                font-weight: bold; letter-spacing: 1px;
            }
            QPushButton:hover { background-color: #2c4a63; }
        """)
        hail_btn.clicked.connect(self.hail_selected_vessel)
        button_layout.addWidget(hail_btn)

        analyze_btn = QPushButton("AI ANALYSIS")
        analyze_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 8px; background-color: #1d3b53;
                color: #64ffda; border: 2px solid #64ffda; border-radius: 3px;
                font-weight: bold; letter-spacing: 1px;
            }
            QPushButton:hover { background-color: #2c4a63; }
        """)
        analyze_btn.clicked.connect(self.ai_analyze_vessel)
        button_layout.addWidget(analyze_btn)

        refresh_btn = QPushButton(" REFRESH")
        refresh_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 8px; background-color: #1d3b53;
                color: #64ffda; border: 2px solid #64ffda; border-radius: 3px;
                font-weight: bold; letter-spacing: 1px;
            }
            QPushButton:hover { background-color: #2c4a63; }
        """)
        refresh_btn.clicked.connect(self.refresh_vessel_list)
        button_layout.addWidget(refresh_btn)

        layout.addLayout(button_layout)

        close_btn = QPushButton("CLOSE CHANNEL")
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
        self.selected_vessel_data = None
        self.refresh_vessel_list()

    def refresh_vessel_list(self):
        """Refresh the list of nearby vessels"""
        nearby = self.controller.get_nearby_ships()
        self.vessel_table.setRowCount(len(nearby))

        for i, ship in enumerate(nearby):
            # Add vessel ID for better tracking
            self.vessel_table.setItem(i, 0, QTableWidgetItem(str(ship.get('id', 'N/A'))))
            self.vessel_table.setItem(i, 1, QTableWidgetItem(ship['vessel_type']))
            self.vessel_table.setItem(i, 2, QTableWidgetItem(f"{ship['distance']:.0f}m"))
            threat = ship['threat_level'] if ship['threat_level'] != 'unknown' else '???'
            self.vessel_table.setItem(i, 3, QTableWidgetItem(threat))
            self.vessel_table.setItem(i, 4, QTableWidgetItem("Active"))

    def on_vessel_selected(self):
        """Handle vessel selection"""
        selected = self.vessel_table.selectedItems()
        if selected:
            row = selected[0].row()
            vessel_id = self.vessel_table.item(row, 0).text()
            vessel_type = self.vessel_table.item(row, 1).text()
            self.selected_vessel_data = {
                'id': vessel_id,
                'type': vessel_type,
                'row': row
            }
            self.message_display.append(f"\n> Selected: {vessel_type} (ID: {vessel_id})")

    def hail_selected_vessel(self):
        """Hail the selected vessel"""
        if not self.selected_vessel_data:
            self.message_display.append("\n[ERROR] No vessel selected.")
            return

        vessel_type = self.selected_vessel_data['type']
        vessel_id = self.selected_vessel_data['id']

        # Enhanced hail with AI scenario consideration
        self.message_display.append(f"\n[OUTGOING] Unidentified vessel {vessel_id}, this is Naval Patrol. Identify yourself immediately.")

        # Simulate response based on AI analysis
        responses = [
            f"This is {vessel_type}. All systems normal, over.",
            f"Roger Naval Patrol, {vessel_type} here. Just conducting routine operations.",
            "...", # Suspicious silence
            f"This is {vessel_type}. We don't need to identify to you.",
        ]

        import random
        response = random.choice(responses)
        self.message_display.append(f"[INCOMING] {response}")

        if response == "...":
            self.message_display.append("\n[AI ALERT] Suspicious behavior detected - vessel not responding!")

    def ai_analyze_vessel(self):
        """AI analysis of selected vessel"""
        if not self.selected_vessel_data:
            self.message_display.append("\n[ERROR] No vessel selected for analysis.")
            return

        vessel_type = self.selected_vessel_data['type']

        # Generate AI analysis based on vessel data
        import random
        threat_indicators = [
            "Unusual speed patterns detected",
            "Course changes suggest evasive behavior", 
            "Signal emissions outside normal parameters",
            "Vessel configuration matches known threat profiles",
            "No anomalies detected - normal civilian traffic"
        ]

        analysis = random.choice(threat_indicators)
        confidence = random.randint(60, 95)

        self.message_display.append(f"\n [AI ANALYSIS] {vessel_type}:")
        self.message_display.append(f"   • {analysis}")
        self.message_display.append(f"   • Confidence Level: {confidence}%")
        self.message_display.append(f"   • Recommended Action: {'MONITOR CLOSELY' if confidence > 80 else 'CONTINUE OBSERVATION'}")

class DistressReportDialog(QDialog):
    """Enhanced Distress report popup with comprehensive threat analysis"""

    def __init__(self, report_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(" DISTRESS CALL - THREAT REPORT ")
        self.setGeometry(200, 100, 900, 700)
        self.setStyleSheet("QDialog { background-color: #0a192f; }")

        layout = QVBoxLayout()

        title = QLabel("DISTRESS CALL INITIATED")
        title.setStyleSheet("""
            font-size: 18px; font-weight: bold; color: #ff4444; letter-spacing: 2px;
            padding: 15px; background-color: #4a0000; border: 2px solid #ff0000;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("AUTOMATED THREAT ANALYSIS & BACKUP REQUEST")
        subtitle.setStyleSheet("""
            font-size: 12px; color: #ff8888; font-weight: bold;
            padding: 5px; text-align: center;
        """)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet("""
            QTextEdit {
                background-color: #0a192f; color: #ff8888;
                font-family: 'Courier New', monospace; font-size: 10px;
                border: 2px solid #ff4444; padding: 10px;
            }
        """)
        self.report_text.setText(report_text)
        layout.addWidget(self.report_text)

        button_layout = QHBoxLayout()

        transmit_btn = QPushButton("TRANSMIT TO COMMAND")
        transmit_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 10px; background-color: #ff4444;
                color: #0a192f; border: none; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #ff6666; }
        """)
        transmit_btn.clicked.connect(self.accept)
        button_layout.addWidget(transmit_btn)

        cancel_btn = QPushButton(" CANCEL DISTRESS")
        cancel_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 10px; background-color: #666666;
                color: #ffffff; border: none; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #888888; }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

class HailVesselDialog(QDialog):
    """Enhanced popup for hailing a vessel with AI integration"""

    def __init__(self, vessel_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"📡 Communication: {vessel_info['vessel_type']}")
        self.setGeometry(400, 300, 600, 350)
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

        # Vessel info section
        info_label = QLabel(f"""
VESSEL DETAILS:
• Type: {vessel_info['vessel_type']}
• Distance: {vessel_info['distance']:.0f}m
• Crew: {vessel_info['crew_count']} personnel
• Scanned: {'Yes' if vessel_info['scanned'] else 'No'}
        """)
        info_label.setStyleSheet("font-size: 10px; color: #8892b0; background-color: #112240; padding: 10px;")
        layout.addWidget(info_label)

        hail_label = QLabel(f"OUTGOING TRANSMISSION:\n {vessel_info['hail_message']}")
        hail_label.setWordWrap(True)
        hail_label.setStyleSheet("font-size: 11px; color: #8892b0; padding: 5px;")
        layout.addWidget(hail_label)

        response_label = QLabel(f"INCOMING RESPONSE:\n{vessel_info['response_message']}")
        response_label.setWordWrap(True)
        response_color = "#ff8888" if vessel_info['is_suspicious'] else "#ccd6f6"
        response_label.setStyleSheet(f"font-size: 12px; color: {response_color}; font-weight: bold; padding: 5px;")
        layout.addWidget(response_label)

        # AI threat assessment
        if vessel_info['is_suspicious']:
            threat_label = QLabel("AI THREAT ASSESSMENT: SUSPICIOUS BEHAVIOR DETECTED")
            threat_label.setStyleSheet("font-size: 11px; color: #ff4444; font-weight: bold; background-color: #4a0000; padding: 8px;")
            layout.addWidget(threat_label)

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
        self.setWindowTitle("Mission Status Log")
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
        log_entries = self.controller.get_status_log()
        self.log_text.setText("\n".join(log_entries))
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

class StatusReportWindow(QDialog):
    """Enhanced status report window"""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Tactical Status Report")
        self.setGeometry(300, 300, 500, 400)
        self.setStyleSheet("QDialog { background-color: #0a192f; }")

        layout = QVBoxLayout()

        title = QLabel("TACTICAL STATUS REPORT")
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
        self.update_report()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_report)
        self.timer.start(1000)

    def update_report(self):
        report = self.controller.get_status_report()
        self.report_text.setText(report)

# =============================================================================
# START MENU
# =============================================================================

class StartMenu(QWidget):
    """Enhanced start menu with beautiful UI"""

    def __init__(self):
        super().__init__()
        self.setStyleSheet("QWidget { background-color: #0a192f; }")

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(50, 50, 50, 50)

        # Title with gradient
        title = QLabel("NAVAL COMBAT SIMULATION")
        title.setStyleSheet("""
            font-size: 28px; font-weight: bold; color: #64ffda; letter-spacing: 3px;
            padding: 20px; background-color: #112240; border: 2px solid #64ffda;
            border-radius: 5px;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Mission selection
        mission_label = QLabel("SELECT MISSION TYPE:")
        mission_label.setStyleSheet("font-size: 14px; color: #64ffda; font-weight: bold;")
        layout.addWidget(mission_label)

        self.mission_combo = QComboBox()
        self.mission_combo.addItems(["Patrol Boat", "Attack Vessel"])
        self.mission_combo.setStyleSheet("""
            QComboBox {
                font-size: 12px; padding: 8px; background-color: #112240;
                color: #64ffda; border: 2px solid #64ffda; border-radius: 3px;
            }
            QComboBox::drop-down { border: none; }
        """)
        layout.addWidget(self.mission_combo)

        # Difficulty selection
        difficulty_label = QLabel("SELECT DIFFICULTY:")
        difficulty_label.setStyleSheet("font-size: 14px; color: #64ffda; font-weight: bold;")
        layout.addWidget(difficulty_label)

        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["Novice", "Intermediate", "Expert"])
        self.difficulty_combo.setStyleSheet("""
            QComboBox {
                font-size: 12px; padding: 8px; background-color: #112240;
                color: #64ffda; border: 2px solid #64ffda; border-radius: 3px;
            }
            QComboBox::drop-down { border: none; }
        """)
        layout.addWidget(self.difficulty_combo)

        layout.addStretch()

        # Start button with hover effect
        self.start_btn = QPushButton("LAUNCH SIMULATION")
        self.start_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; padding: 12px; background-color: #64ffda;
                color: #0a192f; border: none; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #57d8c0; }
        """)
        layout.addWidget(self.start_btn)

        self.setLayout(layout)

# =============================================================================
# SIMULATION WINDOW
# =============================================================================

class SimulationWindow(QMainWindow):
    """Enhanced simulation window with beautiful UI"""

    # In ui.py, inside the SimulationWindow class

    def __init__(self, mission_type: str = "Patrol Boat"):
        super().__init__()
        self.setWindowTitle("Naval Combat Simulation")
        self.resize(1200, 800)
        self.setStyleSheet("QMainWindow { background-color: #0a192f; }")

        self.controller = SimulationController(mission_type=mission_type)
        self.graphics_items = {}
        self.patrol_phase_ui_updated = False

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Radar view with enhanced styling
        self.scene = QGraphicsScene(0, 0, 800, 600)
        gradient = QLinearGradient(0, 0, 0, 600)
        gradient.setSpread(QGradient.Spread.ReflectSpread)
        gradient.setColorAt(0, QColor(10, 25, 47))
        gradient.setColorAt(1, QColor(20, 40, 70))
        self.scene.setBackgroundBrush(QBrush(gradient))

        # Add grid lines
        grid_size = 50
        pen = QPen(QColor(255, 255, 255, 20))  # faint white
        for x in range(0, 801, grid_size):
            self.scene.addLine(x, 0, x, 600, pen)
        for y in range(0, 601, grid_size):
            self.scene.addLine(0, y, 800, y, pen)

        self.view = QGraphicsView(self.scene)
        self.view.setStyleSheet("QGraphicsView { border: none; }")
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.view.mousePressEvent = self.radar_click

        # --- START: FIX ---
        # This tells the radar view to use the main window's key event handlers.
        self.view.keyPressEvent = self.key_press_event
        self.view.keyReleaseEvent = self.key_release_event
        # --- END: FIX ---

        # Main splitter for radar and side panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.view)

        # Side panel with enhanced styling
        side_frame = self.setup_side_panel()
        splitter.addWidget(side_frame)
        splitter.setSizes([800, 400])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStyleSheet("QSplitter::handle { background-color: #64ffda; width: 2px; }")

        main_layout.addWidget(splitter)

        # Bottom panel
        bottom_panel = self.setup_bottom_panel()
        main_layout.addWidget(bottom_panel)

        self.setup_controls()
        self.setup_timer()
        self.update_display()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    """Setup enhanced side panel with beautiful UI"""
       # In ui.py

    # In ui.py

    def setup_side_panel(self):
        """Setup enhanced side panel with beautiful UI"""
        side_frame = QFrame()
        side_frame.setStyleSheet("""
            QFrame {
                background-color: #112240;
                border-left: 2px solid #64ffda;
            }
        """)
        layout = QVBoxLayout(side_frame)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Status display with gradient background
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1d3b53, stop:1 #112240);
                border: 2px solid #64ffda; border-radius: 5px; padding: 10px;
            }
        """)
        status_layout = QVBoxLayout(status_frame)

        self.status_label = QLabel("Status: Approaching Patrol Zone\nUse WASD to navigate")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 12px; color: #64ffda;")
        status_layout.addWidget(self.status_label)
        
        # --- START: FIX ---
        # Create a horizontal layout for contact status
        contact_layout = QHBoxLayout()

        # Create and add the missing enemy_label
        self.enemy_label = QLabel("Contacts: 0")
        self.enemy_label.setStyleSheet("font-size: 11px; color: #ccd6f6; font-weight: bold;")
        contact_layout.addWidget(self.enemy_label)
        
        contact_layout.addStretch() # Add space between labels

        # Create and add the missing hostile_label
        self.hostile_label = QLabel("0 HOSTILE")
        self.hostile_label.setStyleSheet("font-size: 11px; color: #ff4444; font-weight: bold;")
        contact_layout.addWidget(self.hostile_label)

        # Add the new layout to the main status panel
        status_layout.addLayout(contact_layout)
        # --- END: FIX ---

        layout.addWidget(status_frame)

        # Vessel details
        details_frame = QFrame()
        details_frame.setStyleSheet("""
            QFrame {
                background-color: #1d3b53; border: 2px solid #64ffda;
                border-radius: 5px; padding: 10px;
            }
        """)
        details_layout = QVBoxLayout(details_frame)

        details_title = QLabel("VESSEL DETAILS")
        details_title.setStyleSheet("font-size: 14px; color: #64ffda; font-weight: bold;")
        details_layout.addWidget(details_title)

        self.details_label = QLabel("No vessel selected")
        self.details_label.setStyleSheet("font-size: 11px; color: #ccd6f6;")
        self.details_label.setWordWrap(True)
        details_layout.addWidget(self.details_label)

        layout.addWidget(details_frame)

        # Action buttons
        self.setup_action_buttons(layout)

        layout.addStretch()

        # System controls
        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #1d3b53; border: 2px solid #64ffda;
                border-radius: 5px; padding: 10px;
            }
        """)
        control_layout = QVBoxLayout(control_frame)

        control_title = QLabel("SYSTEM CONTROLS")
        control_title.setStyleSheet("font-size: 14px; color: #64ffda; font-weight: bold;")
        control_layout.addWidget(control_title)

        system_layout = QHBoxLayout()
        
        self.pause_btn = QPushButton("▶ START")
        self.pause_btn.setStyleSheet(self.get_system_button_style())
        system_layout.addWidget(self.pause_btn)
        
        log_btn = QPushButton("LOG")
        log_btn.setStyleSheet(self.get_system_button_style())
        log_btn.clicked.connect(self.show_status_log)
        system_layout.addWidget(log_btn)

        report_btn = QPushButton("REPORT")
        report_btn.setStyleSheet(self.get_system_button_style())
        report_btn.clicked.connect(self.show_status_report)
        system_layout.addWidget(report_btn)

        control_layout.addLayout(system_layout)
        layout.addWidget(control_frame)

        layout.addStretch()
        return side_frame

    def setup_action_buttons(self, layout):
        """Setup vessel action buttons"""
        action_frame = QFrame()
        action_layout = QVBoxLayout(action_frame)

        button_style = """
            QPushButton {
                font-size: 11px; padding: 8px; background-color: #1d3b53;
                color: #64ffda; border: 2px solid #64ffda; border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2c4a63; }
            QPushButton:disabled {
                background-color: #333333; color: #666666; border-color: #666666;
            }
        """

        self.intercept_btn = QPushButton(" INTERCEPT")
        self.intercept_btn.setStyleSheet(button_style)
        self.intercept_btn.setEnabled(False)
        self.intercept_btn.clicked.connect(self.intercept_vessel)
        action_layout.addWidget(self.intercept_btn)

        self.mark_safe_btn = QPushButton(" MARK SAFE")
        self.mark_safe_btn.setStyleSheet(button_style)
        self.mark_safe_btn.setEnabled(False)
        self.mark_safe_btn.clicked.connect(self.mark_safe)
        action_layout.addWidget(self.mark_safe_btn)

        self.mark_threat_btn = QPushButton(" MARK THREAT")
        self.mark_threat_btn.setStyleSheet(button_style)
        self.mark_threat_btn.setEnabled(False)
        self.mark_threat_btn.clicked.connect(self.mark_as_threat)
        action_layout.addWidget(self.mark_threat_btn)

        layout.addWidget(action_frame)

    def setup_bottom_panel(self):
        """Setup enhanced bottom panel with communication and distress call"""
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet("""
            QFrame {
                background-color: #112240;
                border: 2px solid #64ffda;
                border-radius: 5px;
            }
        """)
        bottom_frame.setMaximumHeight(150)

        layout = QHBoxLayout(bottom_frame)

        # Communication section (left)
        comm_section = self.setup_communication_section()
        layout.addWidget(comm_section)

        # Distress section (right)
        distress_section = self.setup_distress_section()
        layout.addWidget(distress_section)

        return bottom_frame

    def setup_communication_section(self):
        """Setup communication section in bottom panel"""
        comm_frame = QFrame()
        comm_layout = QVBoxLayout(comm_frame)

        comm_title = QLabel(" COMMUNICATION CENTER")
        comm_title.setStyleSheet("""
            font-size: 14px; font-weight: bold; color: #64ffda;
            padding: 5px; background-color: #1d3b53; border: 1px solid #64ffda;
        """)
        comm_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        comm_layout.addWidget(comm_title)

        # Quick status
        self.comm_status = QLabel("Monitoring communication channels...")
        self.comm_status.setStyleSheet("font-size: 10px; color: #8892b0; padding: 5px;")
        comm_layout.addWidget(self.comm_status)

        # Communication button
        comm_btn = QPushButton(" OPEN COMMUNICATION WINDOW")
        comm_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 10px; background-color: #64ffda;
                color: #0a192f; border: none; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #57d8c0; }
        """)
        comm_btn.clicked.connect(self.open_communication_window)
        comm_layout.addWidget(comm_btn)

        return comm_frame

    def setup_distress_section(self):
        """Setup distress call section in bottom panel"""
        distress_frame = QFrame()
        distress_layout = QVBoxLayout(distress_frame)

        distress_title = QLabel(" EMERGENCY RESPONSE")
        distress_title.setStyleSheet("""
            font-size: 14px; font-weight: bold; color: #ff4444;
            padding: 5px; background-color: #4a0000; border: 1px solid #ff0000;
        """)
        distress_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        distress_layout.addWidget(distress_title)

        # Distress status
        self.distress_status = QLabel("Emergency backup available")
        self.distress_status.setStyleSheet("font-size: 10px; color: #ff8888; padding: 5px;")
        distress_layout.addWidget(self.distress_status)

        # Distress button
        self.distress_btn = QPushButton(" INITIATE DISTRESS CALL")
        self.distress_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 10px; background-color: #ff4444;
                color: #ffffff; border: none; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #ff6666; }
            QPushButton:disabled {
                background-color: #666666; color: #999999;
            }
        """)
        self.distress_btn.setEnabled(False)
        self.distress_btn.clicked.connect(self.initiate_distress_call)
        distress_layout.addWidget(self.distress_btn)

        return distress_frame

    def get_system_button_style(self):
        """Get system button styling"""
        return """
            QPushButton {
                font-size: 10px; padding: 6px; background-color: #1d3b53;
                color: #64ffda; border: 2px solid #64ffda; border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2c4a63; }
        """

    def setup_controls(self):
        """Setup control event handlers"""
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.view.setFocus()

    def setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(50)  # 20 FPS

    def key_press_event(self, event):
        """Handle key press events"""
        key_map = {
            Qt.Key.Key_W: "w", Qt.Key.Key_A: "a",
            Qt.Key.Key_S: "s", Qt.Key.Key_D: "d",
            Qt.Key.Key_Space: "space"
        }

        if event.key() in key_map:
            if key_map[event.key()] == "space":
                self.controller.stop_player_movement()
            else:
                self.controller.set_key_state(key_map[event.key()], True)

        event.accept()

    def key_release_event(self, event):
        """Handle key release events"""
        key_map = {
            Qt.Key.Key_W: "w", Qt.Key.Key_A: "a",
            Qt.Key.Key_S: "s", Qt.Key.Key_D: "d"
        }

        if event.key() in key_map:
            self.controller.set_key_state(key_map[event.key()], False)

        event.accept()

    def open_communication_window(self):
        """Open the enhanced communication window"""
        comm_window = CommunicationWindow(self.controller, self)
        comm_window.exec()

    def initiate_distress_call(self):
        """Enhanced distress call with comprehensive reporting"""
        if not self.controller.selected_unit:
            self.distress_status.setText("No target selected for distress call")
            return

        # Generate comprehensive distress report
        report = self.controller.generate_distress_report()

        # Show distress report dialog
        dialog = DistressReportDialog(report, self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            # Execute distress call
            message = self.controller.distress_call()
            self.distress_status.setText(message)
            self._clear_selection()
            self.update_display()

    def show_status_log(self):
        """Show status log window"""
        log_window = StatusLogWindow(self.controller, self)
        log_window.exec()

    def show_status_report(self):
        """Show status report window"""
        report_window = StatusReportWindow(self.controller, self)
        report_window.exec()

    def update_display(self):
        """Enhanced display update with zone visualization"""
        # Clear existing graphics
        for items in self.graphics_items.values():
            for item in items:
                self.scene.removeItem(item)
        self.graphics_items.clear()

        # Draw patrol zone
        zone = self.controller.get_zone_info()
        zone_item = QGraphicsRectItem(zone["x"], zone["y"], zone["width"], zone["height"])

        # Dynamic zone bordering based on state (no fill, matching default background implicitly)
        if self.controller.is_in_patrol_zone():
            zone_border = QColor(100, 255, 218, 255)  # Green border when inside
        else:
            zone_border = QColor(255, 70, 70, 255)    # Red border when outside

        zone_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        zone_item.setPen(QPen(zone_border, 3, Qt.PenStyle.DashLine))
        self.scene.addItem(zone_item)

        # Draw vessels
        vessels = self.controller.get_vessel_positions()
        player_found = False
        other_vessels = 0

        for vessel in vessels:
            vessel_id = vessel['id']
            is_player = vessel['is_player']

            if is_player:
                player_found = True
                player_items = []
                triangle = QPolygonF([QPointF(0, -15), QPointF(-10, 10), QPointF(10, 10)])
                player_item = QGraphicsPolygonItem(triangle)
                player_item.setBrush(QBrush(QColor(100, 255, 218)))
                player_item.setPen(QPen(QColor(200, 255, 230), 2))
                player_item.setPos(vessel['x'], vessel['y'])
                self.scene.addItem(player_item)
                player_items.append(player_item)
                self.graphics_items[vessel_id] = player_items

            elif not self.controller.is_patrol_phase_active():
                other_vessels += 1
                items = []

                # Enhanced vessel visualization
                if vessel['scanned']:
                    if vessel['threat_level'] == "neutral":
                        color = QColor(100, 255, 218)
                        border = QColor(150, 255, 230)
                    elif vessel['threat_level'] == "possible":
                        color = QColor(255, 255, 255)
                        border = QColor(200, 200, 200)
                    elif vessel['threat_level'] == "confirmed":
                        color = QColor(255, 70, 70)
                        border = QColor(200, 50, 50)
                    else:
                        color = QColor(136, 146, 176)
                        border = QColor(100, 110, 140)
                else:
                    color = QColor(136, 146, 176)
                    border = QColor(100, 110, 140)

                item = QGraphicsEllipseItem(-8, -8, 16, 16)
                item.setBrush(QBrush(color))
                item.setPen(QPen(border, 2))
                item.setPos(vessel['x'], vessel['y'])
                self.scene.addItem(item)
                items.append(item)

                if vessel.get('selected', False):
                    highlight = QGraphicsEllipseItem(-15, -15, 30, 30)
                    highlight.setPen(QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine))
                    highlight.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                    highlight.setPos(vessel['x'], vessel['y'])
                    self.scene.addItem(highlight)
                    items.append(highlight)

                self.graphics_items[vessel_id] = items

        # Update status labels
        status = self.controller.get_status_info()
        self.enemy_label.setText(f"Contacts: {status['total_threats']}")
        self.hostile_label.setText(f"{status['confirmed_threats']} HOSTILE")
        self.update_bottom_panel_status()

    def update_bottom_panel_status(self):
        """Update bottom panel status displays"""
        nearby = self.controller.get_nearby_ships()
        self.comm_status.setText(f"Monitoring {len(nearby)} vessels on communication channels")

        if self.controller.selected_unit:
            self.distress_status.setText("Distress call ready for selected vessel")
        else:
            self.distress_status.setText("Select a vessel to enable emergency response")

    def radar_click(self, event):
        """Enhanced radar click handling"""
        if self.controller.is_patrol_phase_active() or not self.controller.is_in_patrol_zone():
            self._clear_selection()
            return

        scene_pos = self.view.mapToScene(event.pos())
        vessel_info = self.controller.handle_vessel_click(scene_pos.x(), scene_pos.y())

        if vessel_info:
            # Show hail dialog
            dialog = HailVesselDialog(vessel_info, self)
            dialog.exec()

            # Update UI based on vessel info
            distance = vessel_info['distance']
            in_intercept_range = distance <= self.controller.INTERCEPT_RANGE
            is_suspicious = vessel_info['is_suspicious']

            self.intercept_btn.setEnabled(in_intercept_range)
            self.mark_safe_btn.setEnabled(not is_suspicious)
            self.mark_threat_btn.setEnabled(is_suspicious)
            self.distress_btn.setEnabled(True)

            threat_text = vessel_info['threat_level'].capitalize() if vessel_info['scanned'] else "Unknown"
            details = (f"Type: {vessel_info['vessel_type']}\n"
                       f"Threat Level: {threat_text}\n"
                       f"Distance: {distance:.0f} m\n\n"
                       f"Crew Size: {vessel_info['crew_count']}\n"
                       f"COMMUNICATION LOGGED.")

            self.details_label.setText(details)
        else:
            self._clear_selection()

        self.update_display()

    def _clear_selection(self):
        """Clear current selection and disable action buttons"""
        self.details_label.setText("No vessel selected")
        self.intercept_btn.setEnabled(False)
        self.mark_safe_btn.setEnabled(False)
        self.mark_threat_btn.setEnabled(False)
        self.distress_btn.setEnabled(False)

    def intercept_vessel(self):
        """Handle intercept action"""
        result = self.controller.intercept_vessel()
        self._clear_selection()
        self.update_display()

    def mark_safe(self):
        """Handle mark safe action"""
        result = self.controller.mark_safe()
        self._clear_selection()
        self.update_display()

    def mark_as_threat(self):
        """Handle mark threat action"""
        result = self.controller.mark_threat()
        self._clear_selection()
        self.update_display()

    def update_simulation(self):
        """Enhanced main update loop"""
        if not self.controller.is_game_over():
            self.controller.update_simulation()
            status = self.controller.get_status_info()

            # Check if patrol phase just ended
            if not self.patrol_phase_ui_updated and not self.controller.is_patrol_phase_active():
                self.patrol_phase_ui_updated = True
                self.pause_btn.setText("⏸ PAUSE")
                self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatioByExpanding)

            # Enhanced status updates
            if self.controller.is_in_patrol_zone():
                zone_status = " IN PATROL ZONE" if self.controller.zone_expanded else "ZONE CONTRACTED"
                self.status_label.setText(
                    f"Status: {zone_status}\nConfirmed Threats: {status['confirmed_threats']}\n"
                    f"Accuracy: {status['accuracy']:.1%}\nClick vessels to interact"
                )
            else:
                self.status_label.setText(
                    "Status: Outside Patrol Zone\nUse WASD to navigate to the red zone."
                )

            self.update_display()

        if self.controller.is_game_over():
            self.timer.stop()

    def toggle_pause(self):
        """Enhanced pause toggle"""
        if self.controller.is_patrol_phase_active():
            self.controller.unpause()
            self.pause_btn.setText(" PAUSE")
            self.view.setFocus()
            return

        paused = self.controller.toggle_pause()
        if paused:
            self.pause_btn.setText(" RESUME")
        else:
            self.pause_btn.setText(" PAUSE")
        self.view.setFocus()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

class NavalSimApp(QMainWindow):
    """Enhanced main application"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(" Naval Combat Simulation v2.0")
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