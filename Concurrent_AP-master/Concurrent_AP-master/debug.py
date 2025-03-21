from tempfile import NamedTemporaryFile

fh = NamedTemporaryFile('w', delete = False, dir = './',
                                    suffix = '.h5')
hdf5_file = fh.name