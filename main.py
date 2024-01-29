from data_reader.ircadb import *
from data_reader.lits import *

if __name__ == "__main__":
    read_ircadb("dataset/ircadb")
    sample = read_lits("dataset/LITS17")