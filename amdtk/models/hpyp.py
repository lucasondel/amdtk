
"""(Hierarchical) Pitman-Yor Process (HPYP) based language model."""

import numpy as np
from numpy.random import multinomial, gamma, beta, uniform
import random
from .hierarchical_language_model import HierarchicalLanguageModel
from .hierarchical_language_model import EMPTY_CONTEXT


class PitmanYorProcess(object):
    """Pitmay-Yor Process (PYP) based language model.

    Attributes
    ----------
    G0 : float
        Value for the base distribution (we assume a flat base
        distribution).
    d : float
        Discount parameter.
    tables : list of intAdd a customer in the restaurant and serve her dish.
        Count of customers sitting at each table.
    dish_table : dictionary int -> set of int
        Mapping dish (i.e. word) to a set of table indices.
    root_process : :class:`PitmanYorProcess`
        Parent of the PYP if the current PY is part of a hierarchy.
    num_tables : int
        Number of tables allocated. Some of them may be empty.

    Methods
    -------
    removeCustomer(dish)
        Remove a customer eating 'dish'.
    chooseTable(dish)
        Choose a table where 'dish' is served or allocate a new one.
    addCustomer(dish)
        Add a customer in the restaurant and serve her a dish.
    serveDish(dish, remove_customer=False)
        Add a customer to the restaurant.
    predictiveProbability(dish)
        Probability of a dish given the current seating arrangement.

    """

    def __init__(self, discount, concentration, G0=None, root_process=None):
        """Initialize a PitmanYor process (PYP) language model.

        Parameters
        ----------
        discount : float
            Discount parameters of the PYP.
        concentration: float
            Concentration parameter of the PYP.
        G0 : float
            Parameter of the uniform distribution. G0 is only use when
            the PYP is at the top of the hierarchy. Otherwise the root
            process is used for the base distribution.
        root_process : :class:`PitmanYorProcess`
            PYP at the next level of the hierachy. The root process is
            needed in hierachical model to evaluate the base
            distribution. Sampling a PYP will affect its parents PYP.

        """
        self.G0 = G0
        self.d = discount
        self.theta = concentration
        self.tables = np.array([], dtype=int)
        self.dish_table = {}
        self.root_process = root_process

    @property
    def num_tables(self):
        """ Number of tables allocated. """
        return self.tables.shape[0]

    def getWords(self):
        return self.dish_table.keys()

    def removeCustomer(self, dish):
        """Remove a customer eating 'dish'.

        Parameters
        ----------
        dish: object
            identifier of the dish.

        """
        rt_ix = random.sample(self.dish_table[dish], 1)[0]
        self.tables[rt_ix] -= 1
        if self.tables[rt_ix] == 0:
            # Remove the table completely and update maps.
            self.dish_table[dish].remove(rt_ix)
            if self.root_process is not None:
                self.root_process.removeCustomer(dish)
            if not bool(self.dish_table[dish]):
                del self.dish_table[dish]

    def chooseTable(self, dish):
        """Choose a table where 'dish' is served or allocate a new one.

        Parameters
        ----------
        dish: object
            identifier of the dish.

        """
        is_new_table = False
        indices = np.where(self.tables == 0)[0]
        if len(indices) == 0:
            new_table = self.num_tables
        else:
            new_table = indices[0]

        if dish not in self.dish_table:
            if new_table == self.num_tables:
                self.tables = np.append(self.tables, [0])
            self.tables[new_table] += 1
            table = new_table
            is_new_table = True
        else:
            # Probability of the dish from the base distribution.
            if self.root_process is not None:
                G = self.root_process.predictiveProbability(dish)
            else:
                G = self.G0

            dish_tables_ix = list(self.dish_table[dish])
            dish_tables = self.tables[dish_tables_ix]
            Ntd = len(dish_tables)
            pd = np.zeros(len(dish_tables) + 1, dtype=float)
            pd[:Ntd] = dish_tables - self.d
            pd[-1] = (self.theta + (self.d * Ntd)) * G
            pd /= pd.sum()

            assert np.isclose(pd.sum(), 1)

            table = np.where(multinomial(1, pd, size=None) == 1)[0][0]

            if table == Ntd:
                # A new table has been chosen
                if new_table == len(self.tables):
                    self.tables = np.append(self.tables, [0])
                self.tables[new_table] += 1
                table = new_table
                is_new_table = True
            else:
                self.tables[dish_tables_ix[table]] += 1
                table = dish_tables_ix[table]

        return table, is_new_table

    def addCustomer(self, dish):
        """Add a customer in the restaurant and serve her dish.

        Parameters
        ----------
        dish: object
            identifier of the dish.

        """

        table, is_new_table = self.chooseTable(dish)
        try:
            self.dish_table[dish].add(table)
        except KeyError:
            self.dish_table[dish] = set([table])
        return is_new_table

    def serveDish(self, dish, remove_customer=False):
        """ Add a customer to the restaurant.

        Parameters
        ----------
        dish: object
            identifier of the dish.
        remove_customer: boolean
            If true, remove a cutomer eating this particular dish and
            replace her in the restaurant.

        """
        is_new_table = self.addCustomer(dish)
        if is_new_table and self.root_process is not None:
            self.root_process.serveDish(dish, remove_customer=remove_customer)

    def predictiveProbability(self, dish):
        """ Probability of a dish given the current seating arrangement.

        Parameters
        ----------
        dish: object
            identifchooseTableier of the dish.

        Returns
        -------
        p: float
            Predictive probability of the dish.
        """
        Nt = np.count_nonzero(self.tables)
        if dish in self.dish_table:
            dish_tables_ix = list(self.dish_table[dish])
            dish_tables = self.tables[dish_tables_ix]
            Ntd = len(dish_tables)
            ret_val = max(0., (dish_tables.sum() - (self.d*Ntd)))
        else:
            ret_val = 0

        if self.root_process is not None:
            G = self.root_process.predictiveProbability(dish)
        else:
            G = self.G0
        ret_val += (self.theta + (self.d * Nt)) * G

        ret_val /= self.tables.sum() + self.theta

        return ret_val

    def fallbackProbability(self):
        numTables = np.count_nonzero(self.tables)
        numCustomers = self.tables.sum()
        return (self.theta + numTables*self.d) / (self.theta + numCustomers)

    def sampleLogXu(self):
        c_u = self.tables.sum()
        if c_u < 2:
            return 0
        Xu = beta(self.theta + 1, c_u - 1)
        return np.log(Xu)

    def sampleSumYui(self):
        t_u = len(self.tables)
        if t_u < 2:
            # Not a bug: empty term for both sums in calling function!
            return 0, 0
        sum_Yui, sum_one_minus_Yui = 0, 0
        for i in range(1, t_u):
            Yui = uniform() > self.theta/(self.theta + i*self.d)
            sum_Yui += Yui
            sum_one_minus_Yui += 1-Yui
        return sum_Yui, sum_one_minus_Yui

    def sampleSumOneMinusZuwkj(self):
        sum_one_minus_Zuwkj = 0.0
        for dish in self.dish_table:
            dish_tables_ix = list(self.dish_table[dish])
            for table_for_dish in dish_tables_ix:
                c_uwk = self.tables[table_for_dish]
                if c_uwk > 2:
                    for j in range(1, c_uwk):
                        Zuwkj = uniform() < (j -1)/(j - self.d)
                        sum_one_minus_Zuwkj += 1-Zuwkj
        return sum_one_minus_Zuwkj

class HierarchicalPitmanYorProcess(HierarchicalLanguageModel):
    """ Hierarchical Pitmay-Yor Process (HPYP) based language model.

    Attributes
    ----------
    params : list of tuple (discount, concentration)
        Set of parameters for each level of the hierarchy.
    hierarchy : list of dictionary
        Hierarchy of :class:`PitmanYorProcess`.
    order : int
        Order of the language model (i.e. hierarchy).

    Methods
    -------
    __getitem__(key)
        Index operator to access a specific level of the hierarchy.
    addRestaurant(level, context):
        Add a restaurant, i.e. a PYP, for a given context.
    predictiveProbability(self, level, context, dish)
        Probability of a dish given the current seating arrangement.

    """

    def __init__(self, params, G0, vocab=None):
        """ Initialize a Hierarchical PitmanYor process (HPYP) language
        model.

        Parameters
        ----------
        params: list of list/tuple
            Set of parameters (discount, concentration) for each level
            of the hierarchy.
        G0: float
            Parameter of the uniform distribution. This will be the base
            distribution of the root of the hierarchy.
        vocab : list
            List of unique dishes

        """
        self.params = params
        discount, concentration = params[0]
        super(self.__class__, self).__init__(len(params)-1, vocab)
        self.hierarchy[0][EMPTY_CONTEXT] = PitmanYorProcess(*params[0], G0=G0)

    def addRestaurant(self, level, context):
        """Add a restaurant, i.e. a PYP, for a given context.

        No restautant can be added at the top of the hierarchy.

        Parameters
        ----------
        level: int
            Level at which to add the restaurant.
        context: tuple
            Context identifier for the new restaurant.

        Returns
        -------
        rest: :class:`PitmanYorProcess`
            Newly created restaurant.

        """
        assert level > 0, "Cannot add a restaurant at the top of the " \
                          "hierarchy."

        if len(context) > 1:
            new_context = context[1:]
        else:
            new_context = EMPTY_CONTEXT

        discount, concentration = self.params[level]
        try:
            root_process = self.hierarchy[level-1][new_context]
        except KeyError:
            root_process = self.addRestaurant(level-1, new_context)
        restaurant = PitmanYorProcess(discount, concentration,
                                      root_process=root_process)
        self.hierarchy[level][context] = restaurant

        return restaurant

    def predictiveProbability(self, level, context, dish):
        """Probability of a dish given the current seating arrangement.

        Parameters
        ----------
        level: int
            level with which to compute the probability.
        context: object
            identifier of the context of the current dish.
        dish: object
            identifier of the dish.

        Returns
        -------
        p: float
            Predictive probability of the dish given the context.

        """
        try:
            prob = self.hierarchy[level][context].predictiveProbability(dish)
        except KeyError:
            if len(context) > 1:
                new_context = context[1:]
            else:
                new_context = EMPTY_CONTEXT
            prob = self.predictiveProbability(level-1, new_context, dish)

        return prob

    def fallbackProbability(self, level, context):
        return self.hierarchy[level][context].fallbackProbability()

    def resampleAllHyperparameters(self):
        for level in range(len(self.hierarchy)):
            self._resampleHyperparmetersforLevel(level)

    def _resampleHyperparmetersforLevel(self, level):
            a, b, shape, inv_scale = 1, 1, 1, 1
            # accumulate hyper-hyperparameters
            for restaurant in self.hierarchy[level].values():
                sum_Yui, sum_one_minus_Yui = restaurant.sampleSumYui()
                a += sum_Yui
                b += restaurant.sampleSumOneMinusZuwkj()
                shape += sum_one_minus_Yui
                inv_scale -= restaurant.sampleLogXu()

            # resample concentration and discount
            assert (a > 0 and b > 0), \
                "Nonpositive parameters for beta distribution!"
            d= beta(a, b)
            assert (shape > 0 and inv_scale > 0), \
                "Nonpositive parameters for gamma distribution!"
            theta = gamma(shape, 1/inv_scale)

            # update all hyperparameter values in main structure
            # and all restaurants
            self.params[level] = d, theta
            for restaurant in self.hierarchy[level].values():
                restaurant.theta = theta
                restaurant.d = d

    def printOutHyperparameters(self):
        for d, theta in self.params:
            print("['{}', '{}']".format(d, theta))
