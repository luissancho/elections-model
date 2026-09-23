from ..core.worker import SequenceJob


class Update(SequenceJob):

    sequence = {
        'ancla': [
            'stripe_customers', 'stripe_charges', 'stripe_refunds', 'stripe_payouts', 'stripe_transactions'
        ],
        'meteo': [
            'aemet_station_daily'
        ]
    }
    default_action = 'update'
