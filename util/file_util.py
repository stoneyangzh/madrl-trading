def save_to_file(filename, data):
    f = open(filename,"w")
    # write file
    f.write(str(data))
    # close file
    f.close()
def read_file(file_name):
    f = open(file_name, "r")
    return f.read()