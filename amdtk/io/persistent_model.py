"""
Base class for all persitent model.

Copyright (C) 2017, Lucas Ondel

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

"""

import abc
import pickle

class PersistentModel(metaclass=abc.ABCMeta):
    """Base class for classes that can be serialized."""

    @abc.abstractmethod
    def to_dict(self):
        """Get the data of the model into a dictionary.

        Returns
        -------
        dict : dictionary
            Dictionary containing the model's data.

        """
        pass

    @abc.abstractstaticmethod
    def load_from_dict(model_data):
        """Create and initialize the model from a dictionary.

        Parameters
        ----------
        model_data : dictionary
            Dictionary containing the model's data.

        Returns
        -------
        model : :class:`PersistentModel`
            Model initialized from the data.

        """
        pass

    def save(self, file_obj):
        """Store the model.

        Parameters
        ----------
        file_obj : file object
            File-like object  where to store the model.

        """
        retval = self.to_dict()
        retval['class'] = self.__class__
        pickle.dump(retval, file_obj)

    @staticmethod
    def load(file_obj):
        """Load a model.

        Parameters
        ----------
        file_obj : file object
            File-like object  where to store the model.

        Returns
        -------
        model : :class:`PersistentModel`
            Model initialized from the stored data.

        """
        model_data = pickle.load(file_obj)
        cls = model_data['class']
        return cls.load_from_dict(model_data)

