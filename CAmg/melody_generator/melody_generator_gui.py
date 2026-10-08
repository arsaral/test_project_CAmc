# melody_generator_gui.py

import sys
import random

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QTextEdit,
    QMessageBox,
    QComboBox
)

from music21 import stream, note, instrument, tempo, meter


def generate_random_melody(
    block_base,
    block_height,
    duration,
    metronome_number,
    note_count,
    freshness_limit,
    difficulty,
    instrument_name,
    output_path,
):

    base_midi = note.Note(block_base).pitch.midi
    upper_midi = base_midi + block_height

    score = stream.Score()
    part = stream.Part()
    part.id = "Melody"

    instrument_mapping = {
        "Violin": instrument.Violin(),
        "Flute": instrument.Flute(),
        "Piano": instrument.Piano(),
        "Clarinet": instrument.Clarinet(),
        "Trumpet": instrument.Trumpet(),
        "Cello": instrument.Violoncello(),
        "Bassoon": instrument.Bassoon(),
        "Oboe": instrument.Oboe(),
    }

    part.append(
        instrument_mapping.get(
            instrument_name,
            instrument.Violin()
        )
    )

    part.append(
        tempo.MetronomeMark(
            number=metronome_number
        )
    )

    part.append(
        meter.TimeSignature("4/4")
    )

    melody = []
    recent_notes = []
    last_pitch = None

    for _ in range(note_count):

        while True:

            available_notes = [
                n for n in range(
                    base_midi,
                    upper_midi + 1
                )
                if n not in recent_notes
            ]

            chosen_pitch = (
                random.choice(available_notes)
                if available_notes
                else random.randint(
                    base_midi,
                    upper_midi
                )
            )

            if (
                last_pitch is None
                or abs(chosen_pitch - last_pitch)
                < difficulty
            ):
                break

        new_note = note.Note(
            midi=chosen_pitch,
            quarterLength=duration
        )

        melody.append(new_note)

        recent_notes.append(chosen_pitch)

        if len(recent_notes) > freshness_limit:
            recent_notes.pop(0)

        last_pitch = chosen_pitch

    part.append(melody)
    score.append(part)

    score.write("musicxml", output_path)


class MelodyGenerator(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Performable Random Melody Generator"
        )

        self.resize(700, 500)

        self.output_path = ""

        self.create_ui()

    def create_ui(self):

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.base_note = QLineEdit("C4")

        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 127)
        self.height_spin.setValue(12)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setDecimals(2)
        self.duration_spin.setRange(0.125, 8.0)
        self.duration_spin.setValue(0.5)

        self.metro_spin = QSpinBox()
        self.metro_spin.setRange(20, 300)
        self.metro_spin.setValue(120)

        self.note_count_spin = QSpinBox()
        self.note_count_spin.setRange(1, 5000)
        self.note_count_spin.setValue(120)

        self.freshness_spin = QSpinBox()
        self.freshness_spin.setRange(0, 100)
        self.freshness_spin.setValue(5)

        self.difficulty_spin = QSpinBox()
        self.difficulty_spin.setRange(1, 48)
        self.difficulty_spin.setValue(15)

        self.instrument_combo = QComboBox()

        self.instrument_combo.addItems([
            "Violin",
            "Flute",
            "Piano",
            "Clarinet",
            "Trumpet",
            "Cello",
            "Bassoon",
            "Oboe"
        ])

        form.addRow("Base Note:", self.base_note)
        form.addRow("Pitch Height:", self.height_spin)
        form.addRow("Duration:", self.duration_spin)
        form.addRow("Metronome:", self.metro_spin)
        form.addRow("Note Count:", self.note_count_spin)
        form.addRow("Freshness Limit:", self.freshness_spin)
        form.addRow("Difficulty:", self.difficulty_spin)
        form.addRow("Instrument:", self.instrument_combo)

        layout.addLayout(form)

        file_layout = QHBoxLayout()

        self.file_label = QLabel(
            "No output file selected"
        )

        browse_button = QPushButton(
            "Choose Output File"
        )

        browse_button.clicked.connect(
            self.choose_file
        )

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(browse_button)

        layout.addLayout(file_layout)

        self.generate_button = QPushButton(
            "Generate Melody"
        )

        self.generate_button.clicked.connect(
            self.generate
        )

        layout.addWidget(self.generate_button)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addWidget(self.log)

    def choose_file(self):

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save MusicXML",
            "random_melody.musicxml",
            "MusicXML (*.musicxml)"
        )

        if filename:
            self.output_path = filename
            self.file_label.setText(filename)

    def generate(self):

        if not self.output_path:

            QMessageBox.warning(
                self,
                "Missing File",
                "Please select an output file."
            )
            return

        try:

            generate_random_melody(
                block_base=self.base_note.text(),
                block_height=self.height_spin.value(),
                duration=self.duration_spin.value(),
                metronome_number=self.metro_spin.value(),
                note_count=self.note_count_spin.value(),
                freshness_limit=self.freshness_spin.value(),
                difficulty=self.difficulty_spin.value(),
                instrument_name=self.instrument_combo.currentText(),
                output_path=self.output_path
            )

            self.log.append(
                f"Generated: {self.output_path}"
            )

            QMessageBox.information(
                self,
                "Success",
                "MusicXML file generated."
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Error",
                str(e)
            )

            self.log.append(
                f"ERROR: {e}"
            )


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MelodyGenerator()
    window.show()

    window.raise_()
    window.activateWindow()

    sys.exit(app.exec())