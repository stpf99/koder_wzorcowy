import sys
import os
import pickle
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image

class ImageCoder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Coder with Random Generation and Dimensions")
        self.setGeometry(100, 100, 600, 400)
        self.image_data = None
        self.encoded_data = None
        self.decoded_image = None
        self.image_height = None
        self.image_width = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.input_label = QLabel("Brak obrazu wejściowego")
        self.input_label.setAlignment(Qt.AlignCenter)
        self.output_label = QLabel("Brak obrazu wyjściowego")
        self.output_label.setAlignment(Qt.AlignCenter)

        self.generate_button = QPushButton("Generuj losowy obraz")
        self.load_button = QPushButton("Wczytaj obraz")
        self.encode_button = QPushButton("Zakoduj obraz")
        self.save_encoded_button = QPushButton("Zapisz zakodowane dane")  # New button
        self.load_encoded_button = QPushButton("Wczytaj zakodowane dane")  # New button
        self.decode_button = QPushButton("Odkoduj obraz")
        self.save_button = QPushButton("Zapisz odkodowany obraz")

        layout.addWidget(self.input_label)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.load_button)
        layout.addWidget(self.encode_button)
        layout.addWidget(self.save_encoded_button)
        layout.addWidget(self.load_encoded_button)
        layout.addWidget(self.decode_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.output_label)

        self.generate_button.clicked.connect(self.generate_random_image)
        self.load_button.clicked.connect(self.load_image)
        self.encode_button.clicked.connect(self.encode_image)
        self.save_encoded_button.clicked.connect(self.save_encoded_data)
        self.load_encoded_button.clicked.connect(self.load_encoded_data)
        self.decode_button.clicked.connect(self.decode_image)
        self.save_button.clicked.connect(self.save_image)

    def generate_random_image(self):
        # Generate 8x8 random image for simplicity
        self.image_height, self.image_width = 8, 8
        random_data = np.random.randint(0, 2, (self.image_height, self.image_width))
        self.image_data = random_data.flatten()
        image_array = random_data * 255

        image = Image.fromarray(image_array.astype(np.uint8), mode='L')
        random_file = "random_image.jpg"
        image.save(random_file, quality=95)

        qimage = QImage(image_array.astype(np.uint8), self.image_width, self.image_height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage).scaled(200, 200, Qt.KeepAspectRatio)
        self.input_label.setPixmap(pixmap)
        self.input_label.setText(f"Losowy obraz {self.image_width}x{self.image_height}")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Wczytaj obraz", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            image = Image.open(file_name).convert('L')
            self.image_width, self.image_height = image.size
            threshold = 128
            image = image.point(lambda x: 1 if x > threshold else 0, mode='1')
            self.image_data = np.array(image, dtype=int).flatten()

            image_array = self.image_data.reshape(self.image_height, self.image_width) * 255
            qimage = QImage(image_array.astype(np.uint8), self.image_width, self.image_height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage).scaled(200, 200, Qt.KeepAspectRatio)
            self.input_label.setPixmap(pixmap)
            self.input_label.setText(f"Obraz {self.image_width}x{self.image_height}")

    def encode_image(self):
        if self.image_data is None:
            self.output_label.setText("Najpierw wczytaj lub wygeneruj obraz!")
            return

        self.encoded_data = self.encode_board(self.image_data, self.image_width * self.image_height)
        num_ones = sum(self.image_data)
        self.output_label.setText(f"Zakodowano (1-nek: {num_ones}, {self.image_width}x{self.image_height})")

    def save_encoded_data(self):
        if self.encoded_data is None:
            self.output_label.setText("Najpierw zakoduj obraz!")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Zapisz zakodowane dane", "", "Encoded Files (*.enc)")
        if file_name:
            try:
                with open(file_name, 'wb') as f:
                    pickle.dump({
                        'height': self.image_height,
                        'width': self.image_width,
                        'encoded_data': self.encoded_data
                    }, f)
                self.output_label.setText(f"Zapisano zakodowane dane jako {file_name}")
            except Exception as e:
                self.output_label.setText(f"Błąd zapisu: {e}")

    def load_encoded_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Wczytaj zakodowane dane", "", "Encoded Files (*.enc)")
        if file_name:
            try:
                with open(file_name, 'rb') as f:
                    data = pickle.load(f)
                self.image_height = data['height']
                self.image_width = data['width']
                self.encoded_data = data['encoded_data']
                self.output_label.setText(f"Wczytano dane: {self.image_width}x{self.image_height}")
            except Exception as e:
                self.output_label.setText(f"Błąd wczytywania: {e}")

    def decode_image(self):
        if self.encoded_data is None:
            self.output_label.setText("Najpierw zakoduj lub wczytaj zakodowane dane!")
            return

        decoded_data = self.decode_board(self.encoded_data, self.image_width * self.image_height)
        self.decoded_image = decoded_data

        # Validation
        if not np.array_equal(self.image_data, decoded_data) if self.image_data is not None else True:
            num_ones_input = sum(self.image_data) if self.image_data is not None else 'N/A'
            num_ones_decoded = sum(decoded_data)
            self.output_label.setText(
                f"Błąd dekodowania! 1-nek wejście: {num_ones_input}, 1-nek odkodowane: {num_ones_decoded}"
            )
            return

        image_array = np.array(decoded_data, dtype=np.uint8).reshape(self.image_height, self.image_width) * 255
        qimage = QImage(image_array, self.image_width, self.image_height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage).scaled(200, 200, Qt.KeepAspectRatio)
        self.output_label.setPixmap(pixmap)
        self.output_label.setText(f"Odkodowano poprawnie: {self.image_width}x{self.image_height}")

    def save_image(self):
        if self.decoded_image is None:
            self.output_label.setText("Najpierw odkoduj obraz!")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Zapisz obraz", "", "Images (*.png)")
        if file_name:
            image = Image.fromarray(np.array(self.decoded_image, dtype=np.uint8).reshape(self.image_height, self.image_width) * 255)
            image.save(file_name)
            self.output_label.setText(f"Zapisano obraz jako {file_name}")

    def encode_board(self, board, size):
        positions = [i for i in range(size) if board[i] == 1]
        choice_bit = 0
        if len(positions) > size // 2:
            positions = [i for i in range(size) if board[i] == 0]
            choice_bit = 1

        sequences = []
        remaining = positions.copy()

        while remaining:
            best_seq = self.find_best_sequence(remaining, size)
            if best_seq and self.saves_bits(best_seq):
                sequences.append(best_seq)
                self.remove_positions(remaining, best_seq)
            else:
                break

        # Encoding validation
        encoded_positions = set()
        for seq in sequences:
            _, p, k, n = seq
            for i in range(n):
                pos = p + i * k
                if pos < size:
                    encoded_positions.add(pos)
        encoded_positions.update(remaining)
        if sorted(encoded_positions) != sorted(positions):
            print("Błąd kodowania: pozycje się nie zgadzają!")

        output = [self.image_height, self.image_width, choice_bit, len(sequences)]
        for seq in sequences:
            output.append(seq)
        output.append(len(remaining))
        output.extend(remaining)
        return output

    def find_best_sequence(self, positions, size):
        best = None
        max_covered = 0
        for p in sorted(positions):
            for k in range(1, size // 4 + 1):
                covered = self.count_covered(positions, p, k, size)
                if covered > max_covered:
                    max_covered = covered
                    best = (0, p, k, covered)
        return best

    def count_covered(self, positions, p, k, size):
        count = 0
        current = p
        positions_set = set(positions)
        while current in positions_set and current < size:
            count += 1
            current += k
        return count

    def saves_bits(self, seq):
        _, _, _, n = seq
        return n * 6 > 17

    def remove_positions(self, remaining, seq):
        _, p, k, n = seq
        for i in range(n):
            pos = p + i * k
            if pos in remaining:
                remaining.remove(pos)

    def decode_board(self, code, expected_size=None):
        try:
            height, width, choice_bit, num_seqs = code[0], code[1], code[2], code[3]
            if expected_size is None:
                expected_size = height * width
            sequences = code[4:4 + num_seqs]
            num_remaining = code[4 + num_seqs]
            remaining = code[5 + num_seqs:5 + num_seqs + num_remaining]

            positions = set()
            for seq in sequences:
                if len(seq) != 4:
                    raise ValueError("Invalid sequence format")
                _, p, k, n = seq
                for i in range(n):
                    pos = p + i * k
                    if pos < expected_size:
                        positions.add(pos)
                    else:
                        print(f"Warning: Position {pos} out of range")
            for pos in remaining:
                if pos < expected_size:
                    positions.add(pos)
                else:
                    print(f"Warning: Remaining position {pos} out of range")

            # Initialize board based on choice_bit
            board = [0 if choice_bit == 0 else 1] * expected_size
            for p in positions:
                board[p] = 1 if choice_bit == 0 else 0

            # Set dimensions if not already set
            if self.image_height is None or self.image_width is None:
                self.image_height, self.image_width = height, width

            return board
        except Exception as e:
            print(f"Decoding error: {e}")
            return [0] * expected_size if expected_size else [0] * 64

def main():
    app = QApplication(sys.argv)
    window = ImageCoder()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
