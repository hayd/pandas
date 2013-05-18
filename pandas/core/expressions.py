"""
Expressions
-----------

Offer fast expression evaluation through numexpr

"""
import numpy as np
import re

from datetime import datetime, date

'''from pandas import (
    Series, TimeSeries, DataFrame, Panel, Panel4D, Index, MultiIndex, Int64Index
)'''
from pandas.core.index import Index
import pandas.lib as lib

try:
    import numexpr as ne
    _NUMEXPR_INSTALLED = True
except ImportError:  # pragma: no cover
    _NUMEXPR_INSTALLED = False

_USE_NUMEXPR = _NUMEXPR_INSTALLED
_evaluate    = None
_where       = None

# the set of dtypes that we will allow pass to numexpr
_ALLOWED_DTYPES = dict(evaluate = set(['int64','int32','float64','float32','bool']),
                       where    = set(['int64','float64','bool']))

# the minimum prod shape that we will use numexpr
_MIN_ELEMENTS   = 10000

def set_use_numexpr(v = True):
    # set/unset to use numexpr
    global _USE_NUMEXPR
    if _NUMEXPR_INSTALLED:
        _USE_NUMEXPR = v

    # choose what we are going to do
    global _evaluate, _where
    if not _USE_NUMEXPR:
        _evaluate = _evaluate_standard
        _where    = _where_standard
    else:
        _evaluate = _evaluate_numexpr
        _where    = _where_numexpr

def set_numexpr_threads(n = None):
    # if we are using numexpr, set the threads to n
    # otherwise reset
    try:
        if _NUMEXPR_INSTALLED and _USE_NUMEXPR:
            if n is None:
                n = ne.detect_number_of_cores()
            ne.set_num_threads(n)
    except:
        pass


def _evaluate_standard(op, op_str, a, b, raise_on_error=True):
    """ standard evaluation """
    return op(a,b)

def _can_use_numexpr(op, op_str, a, b, dtype_check):
    """ return a boolean if we WILL be using numexpr """
    if op_str is not None:
        
        # required min elements (otherwise we are adding overhead)
        if np.prod(a.shape) > _MIN_ELEMENTS:

            # check for dtype compatiblity
            dtypes = set()
            for o in [ a, b ]:
                if hasattr(o,'get_dtype_counts'):
                    s = o.get_dtype_counts()
                    if len(s) > 1:
                        return False
                    dtypes |= set(s.index)
                elif isinstance(o,np.ndarray):
                    dtypes |= set([o.dtype.name])

            # allowed are a superset
            if not len(dtypes) or _ALLOWED_DTYPES[dtype_check] >= dtypes:
                return True

    return False

def _evaluate_numexpr(op, op_str, a, b, raise_on_error = False):
    result = None

    if _can_use_numexpr(op, op_str, a, b, 'evaluate'):
        try:
            a_value, b_value = a, b
            if hasattr(a_value,'values'):
                a_value = a_value.values
            if hasattr(b_value,'values'):
                b_value = b_value.values
            result = ne.evaluate('a_value %s b_value' % op_str, 
                                 local_dict={ 'a_value' : a_value, 
                                              'b_value' : b_value }, 
                                 casting='safe')
        except (ValueError), detail:
            if 'unknown type object' in str(detail):
                pass
        except (Exception), detail:
            if raise_on_error:
                raise TypeError(str(detail))

    if result is None:
        result = _evaluate_standard(op,op_str,a,b,raise_on_error)

    return result

def _where_standard(cond, a, b, raise_on_error=True):           
    return np.where(cond, a, b)

def _where_numexpr(cond, a, b, raise_on_error = False):
    result = None

    if _can_use_numexpr(None, 'where', a, b, 'where'):

        try:
            cond_value, a_value, b_value = cond, a, b
            if hasattr(cond_value,'values'):
                cond_value = cond_value.values
            if hasattr(a_value,'values'):
                a_value = a_value.values
            if hasattr(b_value,'values'):
                b_value = b_value.values
            result = ne.evaluate('where(cond_value,a_value,b_value)',
                                 local_dict={ 'cond_value' : cond_value,
                                              'a_value' : a_value, 
                                              'b_value' : b_value }, 
                                 casting='safe')
        except (ValueError), detail:
            if 'unknown type object' in str(detail):
                pass
        except (Exception), detail:
            if raise_on_error:
                raise TypeError(str(detail))

    if result is None:
        result = _where_standard(cond,a,b,raise_on_error)

    return result


# turn myself on
set_use_numexpr(True)

def evaluate(op, op_str, a, b, raise_on_error=False, use_numexpr=True):
    """ evaluate and return the expression of the op on a and b

        Parameters
        ----------

        op :    the actual operand
        op_str: the string version of the op
        a :     left operand
        b :     right operand
        raise_on_error : pass the error to the higher level if indicated (default is False),
                         otherwise evaluate the op with and return the results
        use_numexpr : whether to try to use numexpr (default True)
        """

    if use_numexpr:
        return _evaluate(op, op_str, a, b, raise_on_error=raise_on_error)
    return _evaluate_standard(op, op_str, a, b, raise_on_error=raise_on_error)

def where(cond, a, b, raise_on_error=False, use_numexpr=True):
    """ evaluate the where condition cond on a and b

        Parameters
        ----------

        cond : a boolean array
        a :    return if cond is True
        b :    return if cond is False
        raise_on_error : pass the error to the higher level if indicated (default is False),
                         otherwise evaluate the op with and return the results
        use_numexpr : whether to try to use numexpr (default True)
        """

    if use_numexpr:
        return _where(cond, a, b, raise_on_error=raise_on_error)
    return _where_standard(cond, a, b, raise_on_error=raise_on_error)

class Term(object):
    """ create a term object that holds a field, op, and value

        Parameters
        ----------
        field : dict, string term expression, or the field to operate (must be a valid index/column type of DataFrame/Panel)
        op    : a valid op (defaults to '=') (optional)
                >, >=, <, <=, =, != (not equal) are allowed
        value : a value or list of values (required)
        queryables : a kinds map (dict of column name -> kind), or None i column is non-indexable

        Returns
        -------
        a Term object

        Examples
        --------
        Term(dict(field = 'index', op = '>', value = '20121114'))
        Term('index', '20121114')
        Term('index', '>', '20121114')
        Term('index', ['20121114','20121114'])
        Term('index', datetime(2012,11,14))
        Term('major_axis>20121114')
        Term('minor_axis', ['A','B'])

    """

    _ops = ['<=', '<', '>=', '>', '!=', '==', '=']
    _search = re.compile("^\s*(?P<field>\w+)\s*(?P<op>%s)\s*(?P<value>.+)\s*$" % '|'.join(_ops))
    _max_selectors = 31

    def __init__(self, field, op=None, value=None, queryables=None):
        self.field = None
        self.op = None
        self.value = None
        self.q = queryables or dict()
        self.filter = None
        self.condition = None

        # unpack lists/tuples in field
        while(isinstance(field, (tuple, list))):
            f = field
            field = f[0]
            if len(f) > 1:
                op = f[1]
            if len(f) > 2:
                value = f[2]

        # backwards compatible
        if isinstance(field, dict):
            self.field = field.get('field')
            self.op = field.get('op') or '=='
            self.value = field.get('value')

        # passed a term
        elif isinstance(field, Term):
            self.field = field.field
            self.op = field.op
            self.value = field.value

        # a string expression (or just the field)
        elif isinstance(field, basestring):

            # is a term is passed
            s = self._search.match(field)
            if s is not None:
                self.field = s.group('field')
                self.op = s.group('op')
                self.value = s.group('value')

            else:
                self.field = field

                # is an op passed?
                if isinstance(op, basestring) and op in self._ops:
                    self.op = op
                    self.value = value
                else:
                    self.op = '=='
                    self.value = op

        else:
            raise ValueError(
                "Term does not understand the supplied field [%s]" % field)

        # we have valid fields
        if self.field is None or self.op is None or self.value is None:
            raise ValueError("Could not create this term [%s]" % str(self))

        # = vs ==
        if self.op == '=':
            self.op = '=='

        # we have valid conditions
        if self.op in ['>', '>=', '<', '<=']:
            if hasattr(self.value, '__iter__') and len(self.value) > 1:
                raise ValueError("an inequality condition cannot have multiple values [%s]" % str(self))

        if not hasattr(self.value, '__iter__'):
            self.value = [self.value]

        if len(self.q):
            self.eval()

    def __str__(self):
        return "field->%s,op->%s,value->%s" % (self.field, self.op, self.value)

    def __repr__(self):
        return 'Term(field=%s, op=%s, value=%s)' % (repr(self.field), repr(self.op), repr(self.value))

    @property
    def is_valid(self):
        """ return True if this is a valid field """
        return self.field in self.q

    @property
    def is_in_table(self):
        """ return True if this is a valid column name for generation (e.g. an actual column in the table) """
        return self.q.get(self.field) is not None

    @property
    def kind(self):
        """ the kind of my field """
        return self.q.get(self.field)

    def eval(self):
        """ set the numexpr expression for this term """

        if not self.is_valid:
            raise ValueError("query term is not valid [%s]" % str(self))

        # convert values if we are in the table
        if self.is_in_table:
            values = [self.convert_value(v) for v in self.value]
        else:
            values = [[v, v] for v in self.value]

        # equality conditions
        if self.op in ['==', '!=']:

            # our filter op expression
            if self.op == '!=':
                filter_op = lambda axis, values: not axis.isin(values)
            else:
                filter_op = lambda axis, values: axis.isin(values)


            if self.is_in_table:

                # too many values to create the expression?
                if len(values) <= self._max_selectors:
                    self.condition = "(%s)" % ' | '.join(
                        ["(%s %s %s)" % (self.field, self.op, v[0]) for v in values])

                # use a filter after reading
                else:
                    self.filter = (self.field, filter_op, Index([v[1] for v in values]))

            else:

                self.filter = (self.field, filter_op, Index([v[1] for v in values]))

        else:

            if self.is_in_table:

                self.condition = '(%s %s %s)' % (
                    self.field, self.op, values[0][0])

            else:

                raise TypeError("passing a filterable condition to a non-table indexer [%s]" % str(self))

    def convert_value(self, v):
        """ convert the expression that is in the term to something that is accepted by pytables """

        if self.kind == 'datetime64' or self.kind == 'datetime' :
            v = lib.Timestamp(v)
            if v.tz is not None:
                v = v.tz_convert('UTC')
            return [v.value, v]
        elif isinstance(v, datetime) or hasattr(v, 'timetuple') or self.kind == 'date':
            v = time.mktime(v.timetuple())
            return [v, Timestamp(v) ]
        elif self.kind == 'integer':
            v = int(float(v))
            return [v, v]
        elif self.kind == 'float':
            v = float(v)
            return [v, v]
        elif self.kind == 'bool':
            if isinstance(v, basestring):
                v = not str(v).strip().lower() in ["false", "f", "no", "n", "none", "0", "[]", "{}", ""]
            else:
                v = bool(v)
            return [v, v]
        elif not isinstance(v, basestring):
            v = str(v)
            return [v, v]

        # string quoting
        return ["'" + v + "'", v]

class Expression(object):
    pass