import sys

def read_csv_file(filename: str):

    set1 = []
    set2 = []
    set3 = []
    try:

        with open(filename) as f:
            for line in f:
                fields = line.split(', ')
                set1.append(float(fields[0]))
                set2.append(float(fields[1]))

                if len(fields) == 3:
                    set3.append(float(fields[2]))

        if len(set3) is 0:
            set3 = None

        return set1, set2, set3
    except:
        sys.stderr.write("Error: "+filename+" not found\n")
        return None, None

