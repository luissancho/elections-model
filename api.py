#!/usr/bin/env python

from mtpy import mtpy
from mtpy.lib import pages, webapi

mtpy.run()
api = mtpy.api(routes=pages.ROUTES + webapi.ROUTES, not_found=pages.NOT_FOUND)
