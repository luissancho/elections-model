#!/usr/bin/env python

from mtpy import mtpy
from mtpy.lib import webapi

mtpy.run()
api = mtpy.api(routes=webapi.ROUTES)
