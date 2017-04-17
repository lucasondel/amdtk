import amdtk

with open('ploop.bin', 'rb') as fid:
    model = amdtk.PersistentModel.load(fid)
