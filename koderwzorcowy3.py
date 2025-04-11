import sys
import os
import zlib
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import struct
import math

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
        self.test_cases_button = QPushButton("Generuj przypadki testowe")
        self.load_button = QPushButton("Wczytaj obraz")
        self.encode_button = QPushButton("Zakoduj obraz")
        self.save_encoded_button = QPushButton("Zapisz zakodowane dane")
        self.load_encoded_button = QPushButton("Wczytaj zakodowane dane")
        self.decode_button = QPushButton("Odkoduj obraz")
        self.save_button = QPushButton("Zapisz odkodowany obraz")

        layout.addWidget(self.input_label)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.test_cases_button)
        layout.addWidget(self.load_button)
        layout.addWidget(self.encode_button)
        layout.addWidget(self.save_encoded_button)
        layout.addWidget(self.load_encoded_button)
        layout.addWidget(self.decode_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.output_label)

        self.generate_button.clicked.connect(self.generate_random_image)
        self.test_cases_button.clicked.connect(self.generate_test_cases)
        self.load_button.clicked.connect(self.load_image)
        self.encode_button.clicked.connect(self.encode_image)
        self.save_encoded_button.clicked.connect(self.save_encoded_data)
        self.load_encoded_button.clicked.connect(self.load_encoded_data)
        self.decode_button.clicked.connect(self.decode_image)
        self.save_button.clicked.connect(self.save_image)

    def generate_random_image(self):
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

    def generate_test_cases(self):
        self.image_height, self.image_width = 8, 8
        size = self.image_height * self.image_width

        test_patterns = [
            ("all_black", lambda: np.ones((8, 8), dtype=int)),
            ("every_second", lambda: np.array([[1 if (i * 8 + j) % 2 == 0 else 0 for j in range(8)] for i in range(8)])),
            ("every_third", lambda: np.array([[1 if (i * 8 + j) % 3 == 0 else 0 for j in range(8)] for i in range(8)])),
            ("every_fourth", lambda: np.array([[1 if (i * 8 + j) % 4 == 0 else 0 for j in range(8)] for i in range(8)])),
            ("combo_2_4", lambda: np.array([[1 if (i * 8 + j) % 2 == 0 or (i * 8 + j) % 4 == 0 else 0 for j in range(8)] for i in range(8)])),
            ("combo_3_4", lambda: np.array([[1 if (i * 8 + j) % 3 == 0 or (i * 8 + j) % 4 == 0 else 0 for j in range(8)] for i in range(8)])),
        ]

        if not hasattr(self, 'test_case_index'):
            self.test_case_index = 0

        pattern_name, pattern_func = test_patterns[self.test_case_index]
        self.test_case_index = (self.test_case_index + 1) % len(test_patterns)

        image_array = pattern_func()
        self.image_data = image_array.flatten()
        display_array = image_array * 255

        image = Image.fromarray(display_array.astype(np.uint8), mode='L')
        test_file = f"test_{pattern_name}.jpg"
        image.save(test_file, quality=95)

        qimage = QImage(display_array.astype(np.uint8), self.image_width, self.image_height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage).scaled(200, 200, Qt.KeepAspectRatio)
        self.input_label.setPixmap(pixmap)
        self.input_label.setText(f"Przypadek testowy: {pattern_name} {self.image_width}x{self.image_height}")

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
                # Pack the encoded data into a binary format
                binary_data = self.pack_encoded_data(self.image_height, self.image_width, self.encoded_data)
                # Compress with DEFLATE
                compressed_data = zlib.compress(binary_data)
                with open(file_name, 'wb') as f:
                    f.write(compressed_data)
                self.output_label.setText(f"Zapisano zakodowane dane jako {file_name} ({len(compressed_data)} bajtów)")
            except Exception as e:
                self.output_label.setText(f"Błąd zapisu: {e}")

    def load_encoded_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Wczytaj zakodowane dane", "", "Encoded Files (*.enc)")
        if file_name:
            try:
                with open(file_name, 'rb') as f:
                    compressed_data = f.read()
                # Decompress
                binary_data = zlib.decompress(compressed_data)
                # Unpack
                height, width, encoded_data = self.unpack_encoded_data(binary_data)
                self.image_height = height
                self.image_width = width
                self.encoded_data = encoded_data
                self.output_label.setText(f"Wczytano dane: {self.image_width}x{self.image_height}")
            except Exception as e:
                self.output_label.setText(f"Błąd wczytywania: {e}")

    def decode_image(self):
        if self.encoded_data is None:
            self.output_label.setText("Najpierw zakoduj lub wczytaj zakodowane dane!")
            return

        decoded_data = self.decode_board(self.encoded_data, self.image_width * self.image_height)
        self.decoded_image = decoded_data

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
        # Step 1: Determine positions to encode (1s or 0s)
        positions = [i for i in range(size) if board[i] == 1]
        choice_bit = 0
        if len(positions) > size // 2:
            positions = [i for i in range(size) if board[i] == 0]
            choice_bit = 1

        # Step 2: Detect sequences iteratively
        sequences = []
        remaining = positions.copy()

        # Special case: All 1s or all 0s (step k=0)
        if len(remaining) == size:  # All positions are 1s (or 0s if choice_bit=1)
            sequences.append((0, 0, 0, size))
            remaining = []
        elif len(remaining) == 0:  # No positions (all 0s or all 1s if choice_bit=1)
            sequences.append((0, 0, 0, 0))
        else:
            # Try steps k=2 to k=8 on remaining positions
            for k in range(2, 9):  # Check every second, third, ..., up to eighth
                positions_set = set(remaining)
                if not positions_set:
                    break
                # Find sequences with step k
                for p in sorted(positions_set):
                    if p not in positions_set:  # Skip if already removed
                        continue
                    count = self.count_covered(positions_set, p, k, size)
                    if count >= 2:  # Only store sequences with at least 2 positions
                        sequences.append((0, p, k, count))
                        self.remove_positions(remaining, (0, p, k, count))

        # Step 3: Store remaining positions
        output = [choice_bit, len(sequences)]
        for seq in sequences:
            output.append(seq)
        output.append(len(remaining))
        output.extend(remaining)

        return output

    def count_covered(self, positions, p, k, size):
        count = 0
        current = p
        positions_set = set(positions)
        while current in positions_set and current < size:
            count += 1
            current += k
        return count

    def remove_positions(self, remaining, seq):
        _, p, k, n = seq
        for i in range(n):
            pos = p + i * k
            if pos in remaining:
                remaining.remove(pos)

    def pack_encoded_data(self, height, width, encoded_data):
        # Calculate bits needed for positions
        size = height * width
        pos_bits = math.ceil(math.log2(size))  # Bits needed to store a position

        # Binary data buffer
        binary_data = bytearray()

        # Pack height and width (16 bits each)
        binary_data.extend(struct.pack('!HH', height, width))

        # Pack choice_bit (1 bit) and num_sequences (15 bits)
        choice_bit, num_sequences = encoded_data[0], encoded_data[1]
        combined = (choice_bit << 15) | num_sequences
        binary_data.extend(struct.pack('!H', combined))

        # Pack sequences
        for i in range(num_sequences):
            _, p, k, n = encoded_data[2 + i]
            # Pack p, k, n into a bit stream
            # Use pos_bits for p, 4 bits for k (0-8), pos_bits for n
            k_bits = 4
            # Combine into a bit stream
            bits = (p << (k_bits + pos_bits)) | (k << pos_bits) | n
            total_bits = pos_bits + k_bits + pos_bits
            bytes_needed = (total_bits + 7) // 8
            # Pack into bytes
            binary_data.extend(bits.to_bytes(bytes_needed, byteorder='big'))

        # Pack num_remaining (16 bits)
        num_remaining = encoded_data[2 + num_sequences]
        binary_data.extend(struct.pack('!H', num_remaining))

        # Pack remaining positions
        for i in range(num_remaining):
            pos = encoded_data[3 + num_sequences + i]
            binary_data.extend(pos.to_bytes((pos_bits + 7) // 8, byteorder='big'))

        return bytes(binary_data)

    def unpack_encoded_data(self, binary_data):
        pos = 0

        # Unpack height and width
        height, width = struct.unpack('!HH', binary_data[pos:pos+4])
        pos += 4

        # Unpack choice_bit and num_sequences
        combined = struct.unpack('!H', binary_data[pos:pos+2])[0]
        choice_bit = (combined >> 15) & 1
        num_sequences = combined & 0x7FFF
        pos += 2

        # Calculate bits needed for positions
        size = height * width
        pos_bits = math.ceil(math.log2(size))
        k_bits = 4

        # Unpack sequences
        encoded_data = [choice_bit, num_sequences]
        total_bits = pos_bits + k_bits + pos_bits
        bytes_per_seq = (total_bits + 7) // 8
        for _ in range(num_sequences):
            bits = int.from_bytes(binary_data[pos:pos+bytes_per_seq], byteorder='big')
            pos += bytes_per_seq
            p = (bits >> (k_bits + pos_bits)) & ((1 << pos_bits) - 1)
            k = (bits >> pos_bits) & ((1 << k_bits) - 1)
            n = bits & ((1 << pos_bits) - 1)
            encoded_data.append((0, p, k, n))

        # Unpack num_remaining
        num_remaining = struct.unpack('!H', binary_data[pos:pos+2])[0]
        pos += 2
        encoded_data.append(num_remaining)

        # Unpack remaining positions
        bytes_per_pos = (pos_bits + 7) // 8
        for _ in range(num_remaining):
            pos_val = int.from_bytes(binary_data[pos:pos+bytes_per_pos], byteorder='big')
            encoded_data.append(pos_val)
            pos += bytes_per_pos

        return height, width, encoded_data

    def decode_board(self, code, expected_size=None):
        try:
            choice_bit, num_seqs = code[0], code[1]
            if expected_size is None:
                expected_size = self.image_height * self.image_width
            sequences = code[2:2 + num_seqs]
            num_remaining = code[2 + num_seqs]
            remaining = code[3 + num_seqs:3 + num_seqs + num_remaining]

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

            board = [0 if choice_bit == 0 else 1] * expected_size
            for p in positions:
                board[p] = 1 if choice_bit == 0 else 0

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
