from collections import namedtuple
from numpy import identity

from transport import *

#########################
class twiss(dict):
#########################
  "Twiss parameters from madx output (with free choice of select items)"

  def __init__(self, filename='twiss'):
    self.elems = []

    f = open(filename, 'r')

    for line in f:
      # if ("@ " in line and "%le" in line) :    # FIX to take DPP %s
      if "@ " not in line and "@" in line:
        line = replace(line, "@" , "@ ")

      splt = line.split()
      if "@ " in line and "%" in line and "s" not in splt[2]:
        label = splt[1]
        try:
            self[label] = float(splt[3])
        except:
            print "Problem parsing:", line
            print "Going to be parsed as string"
            try:
              self[label] = splt[3].strip('"')
            except:
              print "Problem persits, let's ignore it!"

      elif "@ " in line and "s" in splt[2]:
        label = splt[1].strip(":")
        self[label] = splt[3].strip('"')

      if "* " in line or "*\t" in line:
        labels = splt[1:]
        Elem = namedtuple("Elem", labels)
        print "labels ",len(labels)

      if "$ " in line or "$\t" in line:
        types = splt[1:]

      if "@" not in line and "*" not in line and "$" not in line:
        vals = []
        for j in range(0,len(labels)):
          if "%hd" in types[j]:
            vals.append(int(splt[j]))
          if "%le" in types[j]:
            vals.append(float(splt[j]))
          if "s" in types[j]:
            vals.append(splt[j].strip('"'))
        self.elems.append(Elem(*vals))

    f.close()
    try:
      labels
      types
    except:
      print "From Metaclass: Bad format or empy file ", filename
      print "Leaving Metaclass"
      exit()

######################### 
## Twiss functionality   
#########################
def matrixForElement(e):
  try:
    if e.KEYWORD == "DRIFT":
      r = DRIFT(**e._asdict())
      return r
    if e.KEYWORD == "QUADRUPOLE":
      if e.K1L > 0:
        r = QF(**e._asdict())
      else:
        r = QD(**e._asdict())
      return r
    else:
      return identity(6)
  except:
    print "The Twiss object doesn't have the desired structure"
    exit()
