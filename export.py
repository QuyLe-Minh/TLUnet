import csv

class BenchmarkData:
    def __init__(self):
        self.data = []
        self.headers = ['Title', 'Params', 'FLOPs', 'Spl', 'RKid', 'LKid', 'Gal', 'Liv', 'Sto', 'Aor', 'Pan', 'Dsc', 'HD95']  # Define appropriate headers

    def add_data(self, record):
        """Add a record to the data list."""
        self.data.append(record)

    def read_from_csv(self, filename):
        """Read data from a CSV file and store it in the list."""
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            self.headers = next(reader)  # Read headers
            self.data = [row for row in reader]

    def export_to_csv(self, filename):
        """Export the data to a CSV file."""
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.headers)  # Write headers
            writer.writerows(self.data)

# Example usage:
benchmark = BenchmarkData()
benchmark.read_from_csv('benchmarking.csv')
benchmark.add_data(['new_value1', 'new_value2', 'new_value3'])
benchmark.export_to_csv('benchmarking.csv')