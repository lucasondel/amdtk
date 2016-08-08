import amdtk
import sys
import os

def main():
  if(len(sys.argv[1:]) < 2):
    sys.stderr.write("Not enough input arguments")
    sys.exit(1)
  
  feature_dir = sys.argv[1]
  segments_file = sys.argv[2]
  
  os.mkdirs(feature_dir + "/tmp") 
  
  with open(segments_file,"r") as fs:
    for line in fs:

      line_vals = line.strip().split(" ")
      time_start = float(line_vals[2])
      time_end = float(line_vals[3])
      key = " ".join(line_vals[0].split(" ")[0:-1])
      file_in = feature_dir + "/" + key + ".fea"
      file_out = feature_dir + "/tmp/" + line_vals[0] + ".fea"          
      x = amdtk.readHtk(file_in)
       


if __name__ == "__main__":
  main()



