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
import json

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

        # Classify the image to choose encoding strategy
        strategy, params = self.classify_image(self.image_data, self.image_width, self.image_height)
        self.encoded_data = self.encode_board(self.image_data, self.image_width * self.image_height, strategy, params)
        self.encoding_strategy = strategy
        self.encoding_params = params
        num_ones = sum(self.image_data)
        self.output_label.setText(f"Zakodowano (1-nek: {num_ones}, {self.image_width}x{self.image_height}, strategia: {strategy})")

    def save_encoded_data(self):
        if self.encoded_data is None:
            self.output_label.setText("Najpierw zakoduj obraz!")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Zapisz zakodowane dane", "", "Encoded Files (*.enc)")
        if file_name:
            try:
                # Create dictionary of encoding parameters
                encoding_dict = {
                    'strategy': self.encoding_strategy,
                    'params': self.encoding_params
                }
                dict_bytes = json.dumps(encoding_dict).encode('utf-8')
                dict_len = len(dict_bytes)

                # Pack the encoded data
                binary_data = self.pack_encoded_data(self.image_height, self.image_width, self.encoded_data)
                
                # Combine dictionary and data
                combined_data = struct.pack('!I', dict_len) + dict_bytes + binary_data
                # Compress with DEFLATE
                compressed_data = zlib.compress(combined_data)
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
                combined_data = zlib.decompress(compressed_data)
                # Extract dictionary
                dict_len = struct.unpack('!I', combined_data[:4])[0]
                encoding_dict = json.loads(combined_data[4:4+dict_len].decode('utf-8'))
                binary_data = combined_data[4+dict_len:]
                # Unpack
                height, width, encoded_data = self.unpack_encoded_data(binary_data)
                self.image_height = height
                self.image_width = width
                self.encoded_data = encoded_data
                self.encoding_strategy = encoding_dict['strategy']
                self.encoding_params = encoding_dict['params']
                self.output_label.setText(f"Wczytano dane: {self.image_width}x{self.image_height}, strategia: {self.encoding_strategy}")
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

    def classify_image(self, board, width, height):
        size = width * height
        num_ones = sum(board)
        image_2d = np.array(board).reshape(height, width)

        # Features for classification
        density = num_ones / size
        # Average run length in rows
        row_runs = []
        for row in image_2d:
            run = 0
            for val in row:
                if val == 1:
                    run += 1
                elif run > 0:
                    row_runs.append(run)
                    run = 0
            if run > 0:
                row_runs.append(run)
        avg_run_length = np.mean(row_runs) if row_runs else 0

        # Check for alternating patterns
        alternating_scores = []
        for k in range(2, 9):
            score = sum(1 for i in range(size) if board[i] == 1 and i % k == 0)
            alternating_scores.append(score / num_ones if num_ones > 0 else 0)

        # Check for blocky patterns
        block_score = 0
        for i in range(height - 2):
            for j in range(width - 2):
                block = image_2d[i:i+3, j:j+3]
                if np.all(block == 1):
                    block_score += 1

        # Simple decision tree
        if density == 1 or density == 0:
            return "uniform", {"steps": [0]}
        elif max(alternating_scores) > 0.8:
            best_k = alternating_scores.index(max(alternating_scores)) + 2
            return "alternating", {"steps": [best_k]}
        elif block_score > (height * width) // 10:
            return "blocky", {"block_size": 3}
        else:
            return "random", {"steps": list(range(2, 9))}

    def encode_board(self, board, size, strategy, params):
        positions = [i for i in range(size) if board[i] == 1]
        choice_bit = 0
        if len(positions) > size // 2:
            positions = [i for i in range(size) if board[i] == 0]
            choice_bit = 1

        sequences = []
        remaining = positions.copy()
        image_2d = np.array(board).reshape(self.image_height, self.image_width)

        if strategy == "uniform":
            if len(remaining) == size:
                sequences.append((0, 0, 0, size))
                remaining = []
            elif len(remaining) == 0:
                sequences.append((0, 0, 0, 0))
        elif strategy == "alternating":
            steps = params["steps"]
            for k in steps:
                positions_set = set(remaining)
                if not positions_set:
                    break
                for p in sorted(positions_set):
                    if p not in positions_set:
                        continue
                    count = self.count_covered(positions_set, p, k, size)
                    if count >= 2:
                        sequences.append((0, p, k, count))
                        self.remove_positions(remaining, (0, p, k, count))
        elif strategy == "blocky":
            block_size = params["block_size"]
            for i in range(self.image_height - block_size + 1):
                for j in range(self.image_width - block_size + 1):
                    block = image_2d[i:i+block_size, j:j+block_size]
                    if np.all(block == (1 if choice_bit == 0 else 0)):
                        start_pos = i * self.image_width + j
                        # Encode as a block (type 1 indicates block)
                        sequences.append((1, start_pos, block_size, block_size))
                        for di in range(block_size):
                            for dj in range(block_size):
                                pos = (i + di) * self.image_width + (j + dj)
                                if pos in remaining:
                                    remaining.remove(pos)
            # After blocks, try alternating patterns
            for k in range(2, 9):
                positions_set = set(remaining)
                if not positions_set:
                    break
                for p in sorted(positions_set):
                    if p not in positions_set:
                        continue
                    count = self.count_covered(positions_set, p, k, size)
                    if count >= 2:
                        sequences.append((0, p, k, count))
                        self.remove_positions(remaining, (0, p, k, count))
        else:  # Random
            # Try diagonals
            positions_set = set(remaining)
            for p in sorted(positions_set):
                row, col = divmod(p, self.image_width)
                if row + 1 < self.image_height and col + 1 < self.image_width:
                    count = self.count_covered(positions_set, p, self.image_width + 1, size)
                    if count >= 2:
                        sequences.append((0, p, self.image_width + 1, count))
                        self.remove_positions(remaining, (0, p, self.image_width + 1, count))
            # Then try alternating patterns
            for k in range(2, 9):
                positions_set = set(remaining)
                if not positions_set:
                    break
                for p in sorted(positions_set):
                    if p not in positions_set:
                        continue
                    count = self.count_covered(positions_set, p, k, size)
                    if count >= 2:
                        sequences.append((0, p, k, count))
                        self.remove_positions(remaining, (0, p, k, count))

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
        type_id, p, k, n = seq
        if type_id == 0:  # Linear sequence
            for i in range(n):
                pos = p + i * k
                if pos in remaining:
                    remaining.remove(pos)
        elif type_id == 1:  # Block
            block_size = k  # k and n are block height and width
            row, col = divmod(p, self.image_width)
            for i in range(block_size):
                for j in range(block_size):
                    pos = (row + i) * self.image_width + (col + j)
                    if pos in remaining:
                        remaining.remove(pos)

    def pack_encoded_data(self, height, width, encoded_data):
        size = height * width
        pos_bits = math.ceil(math.log2(size))

        binary_data = bytearray()
        binary_data.extend(struct.pack('!HH', height, width))

        choice_bit, num_sequences = encoded_data[0], encoded_data[1]
        combined = (choice_bit << 15) | num_sequences
        binary_data.extend(struct.pack('!H', combined))

        for i in range(num_sequences):
            type_id, p, k, n = encoded_data[2 + i]
            # type_id (1 bit), p (pos_bits), k (8 bits for step or block height), n (8 bits for count or block width)
            k_bits = 8
            n_bits = 8
            bits = (type_id << (pos_bits + k_bits + n_bits)) | (p << (k_bits + n_bits)) | (k << n_bits) | n
            total_bits = 1 + pos_bits + k_bits + n_bits
            bytes_needed = (total_bits + 7) // 8
            binary_data.extend(bits.to_bytes(bytes_needed, byteorder='big'))

        num_remaining = encoded_data[2 + num_sequences]
        binary_data.extend(struct.pack('!H', num_remaining))

        for i in range(num_remaining):
            pos = encoded_data[3 + num_sequences + i]
            binary_data.extend(pos.to_bytes((pos_bits + 7) // 8, byteorder='big'))

        return bytes(binary_data)

    def unpack_encoded_data(self, binary_data):
        pos = 0
        height, width = struct.unpack('!HH', binary_data[pos:pos+4])
        pos += 4

        combined = struct.unpack('!H', binary_data[pos:pos+2])[0]
        choice_bit = (combined >> 15) & 1
        num_sequences = combined & 0x7FFF
        pos += 2

        size = height * width
        pos_bits = math.ceil(math.log2(size))
        k_bits = 8
        n_bits = 8

        encoded_data = [choice_bit, num_sequences]
        total_bits = 1 + pos_bits + k_bits + n_bits
        bytes_per_seq = (total_bits + 7) // 8
        for _ in range(num_sequences):
            bits = int.from_bytes(binary_data[pos:pos+bytes_per_seq], byteorder='big')
            pos += bytes_per_seq
            type_id = (bits >> (pos_bits + k_bits + n_bits)) & 1
            p = (bits >> (k_bits + n_bits)) & ((1 << pos_bits) - 1)
            k = (bits >> n_bits) & ((1 << k_bits) - 1)
            n = bits & ((1 << n_bits) - 1)
            encoded_data.append((type_id, p, k, n))

        num_remaining = struct.unpack('!H', binary_data[pos:pos+2])[0]
        pos += 2
        encoded_data.append(num_remaining)

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
                type_id, p, k, n = seq
                if type_id == 0:  # Linear sequence
                    for i in range(n):
                        pos = p + i * k
                        if pos < expected_size:
                            positions.add(pos)
                        else:
                            print(f"Warning: Position {pos} out of range")
                elif type_id == 1:  # Block
                    block_height, block_width = k, n
                    row, col = divmod(p, self.image_width)
                    for i in range(block_height):
                        for j in range(block_width):
                            pos = (row + i) * self.image_width + (col + j)
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
