// Entry module of the notice of the measurement cookies: every page loads it first when the site has a Tag Manager
// container configured (`WEB_GTM_ID`); without one the pages do not carry it.
import {wireConsent} from '../consent.js';

wireConsent();
