import csv


class CSVOperator:
    def __init__(self, file_name, r_w):
        self.path = file_name
        self.file = open(file_name, r_w, newline="")
        if r_w == 'w':
            self.writer = csv.writer(self.file)
        if r_w == 'r':
            self.reader = csv.reader(self.file)

    def write_row(self, lines):
        for line in lines:
            self.writer.writerow(line)

    def end(self):
        self.file.close()

    def get_row_length(self):

        return len(self.reader[0])
    
    def get_row_count(self):
        count = 0
        for row in self.reader:
            count += 1
            count2 = len(row)
        return count, count2
