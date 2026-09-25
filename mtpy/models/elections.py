from ..core.data import Model


class Parties(Model):

    table = 'elections.parties'

    autokey = True

    meta = {
        'id': ['int', 3],
        'name': ['str', None, False],
        'fullname': 'str',
        'parent_id': ['int', 3],
        'color': ['str', 10],
        'block': ['cat', 20],
        'regional': 'bin'
    }


class Pollsters(Model):

    table = 'elections.pollsters'

    autokey = True

    meta = {
        'id': ['int', 3],
        'name': ['str', None, False],
        'mtype': ['cat', 20],
        'quality': 'num',
        'rating': 'num'
    }


class Sponsors(Model):

    table = 'elections.sponsors'

    autokey = True

    meta = {
        'id': ['int', 3],
        'name': ['str', None, False]
    }


class Provinces(Model):

    table = 'elections.provinces'

    autokey = True

    meta = {
        'id': ['int', 3],
        'country': ['cat', 5],
        'name': ['str', None, False],
        'slug': 'str',
        'ncode': ['cat', 10],
        'scode': ['cat', 10],
        'seats': ['int', 2]
    }


class PollstersRatings(Model):

    table = 'elections.pollsters_ratings'

    key = ['event_date', 'event_scope', 'pollster_id']
    sort = ['event_date', 'event_scope', 'pollster_id']

    meta = {
        'event_date': 'dtd',
        'event_scope': 'cat',
        'pollster_id': ['int', 3],
        'quality': 'num',
        'num_events': ['int', 2],
        'num_polls': ['int', 2],
        'num_polls_w': 'num',
        'error_avg': 'num',
        'error_blocks': 'num',
        'error_within': 'num',
        'bias_avg': 'num',
        'bias_blocks': 'num',
        'bias_within': 'num',
        'bias': 'num',
        'bias_dev_adj': 'num',
        'bias_dev_err': 'num',
        'rating_adj': 'num',
        'rating': 'num',
        'weight_rating': 'num'
    }


class PollstersParties(Model):
    """
    House effects: deviation of each pollster for each party in each election (see
    `Computer.compute_house_effects`), percentage points. `dev_result` is the mean signed error of its polls
    of the last `n_days` against the official result (`dev_result_c` centred across pollsters, `industry` the
    industry-wide mean it was centred with); `dev_cycle` its deviation from the consensus of the cycle.
    """

    table = 'elections.pollsters_parties'

    key = ['event_date', 'event_scope', 'pollster_id', 'party_id']
    sort = ['event_date', 'event_scope', 'pollster_id', 'party_id']

    meta = {
        'event_date': 'dtd',
        'event_scope': 'cat',
        'pollster_id': ['int', 3],
        'party_id': ['int', 3],
        'pollster': 'cat',
        'party': 'cat',
        'level': ['num', [10, 4]],
        'n_result': ['int', 2],
        'dev_result': ['num', [10, 4]],
        'dev_result_err': ['num', [10, 4]],
        'dev_result_c': ['num', [10, 4]],
        'industry': ['num', [10, 4]],
        'n_cycle': ['int', 2],
        'dev_cycle': ['num', [10, 4]],
        'dev_cycle_err': ['num', [10, 4]]
    }


class Drift(Model):
    """
    Drift of the fitted poll average of each party over `horizon` days, per past election (see
    `Computer.get_drift_data`): `n` increments `mu(t + horizon) - mu(t)`, their root mean square `rms` and
    mean `bias` (percentage points), and the mean `level` of the party over the cycle.
    """

    table = 'elections.drift'

    key = ['event_date', 'event_scope', 'party_id', 'horizon']
    sort = ['event_date', 'event_scope', 'party_id', 'horizon']

    meta = {
        'event_date': 'dtd',
        'event_scope': 'cat',
        'party_id': ['int', 3],
        'horizon': ['int', 2],
        'party': 'cat',
        'level': ['num', [10, 4]],
        'n': ['int', 2],
        'rms': ['num', [10, 4]],
        'bias': ['num', [10, 4]]
    }


class Polls(Model):

    table = 'elections.polls'

    key = ['event_date', 'event_scope', 'date', 'pollster_id', 'sponsor_id']
    sort = ['event_date', 'event_scope', 'date', 'pollster_id', 'sponsor_id']

    meta = {
        'event_date': 'dtd',
        'event_scope': 'cat',
        'date': 'dtd',
        'pollster_id': ['int', 3],
        'sponsor_id': ['int', 3],
        'pollster': 'cat',
        'sponsor': 'cat',
        'name': 'str',
        'start_date': 'dtd',
        'end_date': 'dtd',
        'pub_date': 'dtd',
        'mtype': ['cat', 20],
        'ctype': ['cat', 20],
        'computed': 'bin',
        'featured': 'bin',
        'sample_size': 'int',
        'proc_sample': 'int',
        'days': 'int',
        'parties': ['int', 2],
        'rating': 'num',
        'error_avg': 'num',
        'error_blocks': 'num',
        'error_within': 'num',
        'bias_avg': 'num',
        'bias_blocks': 'num',
        'bias_within': 'num',
        'bias': 'num',
        'bias_dev_adj': 'num',
        'bias_dev_err': 'num',
        'weight_sample': 'num',
        'weight_over': 'num',
        'weight_rating': 'num',
        'notes': 'str',
        'url': 'str'
    }


class PollsResults(Model):

    table = 'elections.polls_results'

    key = ['event_date', 'event_scope', 'date', 'pollster_id', 'sponsor_id', 'party_id']
    sort = ['event_date', 'event_scope', 'date', 'pollster_id', 'sponsor_id', 'party_id']

    meta = {
        'event_date': 'dtd',
        'event_scope': 'cat',
        'date': 'dtd',
        'pollster_id': ['int', 3],
        'sponsor_id': ['int', 3],
        'party_id': ['int', 3],
        'party': 'cat',
        'pct': 'num',
        'seats': 'int',
        'seats_min': 'int',
        'seats_max': 'int',
        'error': 'num',
        'bias': 'num'
    }


class Forecasts(Model):

    table = 'elections.forecasts'

    key = ['event_date', 'event_scope', 'name']
    sort = ['event_date', 'event_scope', 'name']

    meta = {
        'event_date': 'dtd',
        'event_scope': 'cat',
        'name': ['str', None, False],
        'bmap': 'obj',
        'reg_params': 'obj',
        'parties': ['int', 2],
        'error': 'num',
        'bias': 'num'
    }


class ForecastsResults(Model):

    table = 'elections.forecasts_results'

    key = ['event_date', 'event_scope', 'name', 'region_id', 'party_id']
    sort = ['event_date', 'event_scope', 'name', 'region_id', 'party_id']

    meta = {
        'event_date': 'dtd',
        'event_scope': 'cat',
        'name': ['str', None, False],
        'region_id': ['int', 3],
        'party_id': ['int', 3],
        'region': 'cat',
        'party': 'cat',
        'pct': 'num',
        'seats': 'int',
        'seats_min': 'int',
        'seats_max': 'int',
        'error': 'num',
        'bias': 'num'
    }


class Events(Model):

    table = 'elections.events'

    key = ['date', 'scope']
    sort = ['date', 'scope']

    meta = {
        'date': 'dtd',
        'scope': 'cat',
        'name': ['str', None, False],
        'featured': 'bin'
    }


class EventsData(Model):

    table = 'elections.events_data'

    key = ['date', 'scope', 'region_id']
    sort = ['date', 'scope', 'region_id']

    meta = {
        'date': 'dtd',
        'scope': 'cat',
        'region_id': ['int', 3],
        'region': 'cat',
        'seats': 'int',
        'population': 'int',
        'stations': 'int',
        'registered': 'int',
        'counted': 'int',
        'votes': 'int',
        'abstentions': 'int',
        'blank': 'int',
        'invalid': 'int'
    }


class EventsResults(Model):

    table = 'elections.events_results'

    key = ['date', 'scope', 'region_id', 'party_id']
    sort = ['date', 'scope', 'region_id', 'party_id']

    meta = {
        'date': 'dtd',
        'scope': 'cat',
        'region_id': ['int', 3],
        'party_id': ['int', 3],
        'region': 'cat',
        'party': 'cat',
        'votes': 'int',
        'pct': 'num',
        'seats': 'int'
    }
