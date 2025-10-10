import logging
from datetime import timedelta
from typing import Any, Optional, Union

import jwt
import requests

from fhir_pyrate.util import now_utc

logger = logging.getLogger(__name__)


class TokenAuth(requests.auth.AuthBase):
    """
    Performs token authentication and handles token refreshes. This class first performs simple
    BasicAuth authentication to obtain a token, and then appends this token to all its requests.
    The token can also be refreshed using a refresh URL.
    The session has a hook that makes sure that the token stays up to date. If the token is a JWT
    token, it is decoded, and we check if the validity of the token is about to be revoked
    (if 75% of the time has already passed). If the token is not a JWT token, it is possible to
    specify a token_refresh_delta (as a timedelta object or as an integer amount of minutes), and
    the token will be refreshed using the refresh URL within that interval.

    :param username: The username of the user
    :param password: The password of the user
    :param auth_url: The URL where the user can be authenticated
    :param refresh_url: A possible refresh URL to get a new token
    :param session: The requests.Session that should be authenticated
    :param max_login_attempts: The maximum number of logins that can be performed
    :param token_refresh_delta: Either a timedelta object that tells us how often the token
    should be refreshed, or a number of minutes; this does not need to be specified for JWT tokens
    that contain the expiry date
    :param jwt_refresh_leeway: Either a timedelta object or a number of minutes to refresh
    a JWT token before its exp time. If provided and the token has an exp claim, the token
    will be refreshed when now >= (exp - leeway). If not provided, a fallback rule refreshes
    after 75% of the token lifetime has elapsed (if iat and exp are present).
    """

    def __init__(
        self,
        username: str,
        password: str,
        auth_url: str,
        refresh_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
        max_login_attempts: int = 5,
        token_refresh_delta: Optional[Union[int, timedelta]] = None,
        jwt_refresh_leeway: Optional[Union[int, timedelta]] = None,
    ) -> None:
        self._username = username
        self._password = password
        # Future authenticated session
        if session is None:
            self._session = requests.Session()
        else:
            self._session = session
        # Session for handling the tokens
        self._token_session = requests.Session()
        self.auth_url = auth_url
        self.refresh_url = refresh_url
        self._max_login_attempts = max_login_attempts
        self._token_refresh_delta = (
            token_refresh_delta
            if isinstance(token_refresh_delta, timedelta)
            else timedelta(minutes=token_refresh_delta)
            if token_refresh_delta is not None
            else None
        )
        self._jwt_refresh_leeway: Optional[timedelta] = None
        # Interpret int leeway as minutes for consistency with token_refresh_delta
        if isinstance(jwt_refresh_leeway, timedelta):
            self._jwt_refresh_leeway = jwt_refresh_leeway
        elif jwt_refresh_leeway is not None:
            self._jwt_refresh_leeway = timedelta(minutes=jwt_refresh_leeway)  # type: ignore[arg-type]
        self.token: Optional[str] = None
        self._authenticate()
        self.auth_time = now_utc()

    def _authenticate(self) -> None:
        """
        Authenticate the user using the authentication URL and sets the token.
        """
        # Authentication to get the token
        response = self._token_session.get(
            f"{self.auth_url}", auth=(self._username, self._password)
        )
        response.raise_for_status()
        self.token = response.text

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        """
        Set the necessary authentication header of the current request.

        :param r: The prepared request that should be sent
        :return: The prepared request
        """
        # Proactively refresh the token before each request, if needed
        if self.token is None or self.is_refresh_required():
            try:
                self.refresh_token()
            except Exception:  # Let the request fail later if refresh also fails
                logger.exception("Token refresh attempt failed before request send.")
        r.headers.update({"Authorization": f"Bearer {self.token}"})
        return r

    def is_refresh_required(self) -> bool:
        """
        Compute whether the token should be refreshed according to the given token and to the
        _token_refresh_delta variable.

        :return: Whether the token is about to expire and should thus be refreshed
        """
        # If the token is currently None, then it should always be refreshed
        if self.token is None:
            return True
        try:
            decoded = jwt.decode(
                jwt=self.token,
                options={"verify_signature": False},
            )
            now_ts = now_utc().timestamp()
            exp = decoded.get("exp")
            iat = decoded.get("iat")
            # If exp is present, prefer a leeway-based refresh if configured
            if exp is not None:
                if self._jwt_refresh_leeway is not None:
                    leeway_seconds = self._jwt_refresh_leeway.total_seconds()
                    return now_ts >= (exp - leeway_seconds)
                # Fallback: refresh once the last 25% of token lifetime begins
                if iat is not None:
                    refresh_interval = (exp - iat) / 4
                    return now_ts > (exp - refresh_interval)
                # If iat missing but exp exists, refresh only at expiry (no early refresh)
                return now_ts >= exp
            # No exp claim; fall back to non-JWT/counter-based logic below
            return (
                self._token_refresh_delta is not None
                and (now_utc() - self.auth_time) > self._token_refresh_delta
            )
        except jwt.exceptions.PyJWTError:
            # If we are here it means that it is not a JWT token
            # If no user limit has been specified, then we do not refresh
            # If it has been specified and the time is almost run out
            return (
                self._token_refresh_delta is not None
                and (now_utc() - self.auth_time) > self._token_refresh_delta
            )

    def refresh_token(self, token: Optional[str] = None) -> None:
        """
        Refresh the current session either by logging in again or by refreshing the token.

        :param token: If a refresh URL has not been provided, a new token can be provided here as
        parameter
        """
        logger.info("Refreshing session...")
        if token is not None:
            self.token = token
            self.auth_time = now_utc()
        elif self.refresh_url is not None:
            response = self._token_session.get(f"{self.refresh_url}")
            # Was not refreshed on time
            if response.status_code == requests.codes.unauthorized:
                self._authenticate()
            else:
                response.raise_for_status()
                self.token = response.text
            self.auth_time = now_utc()
        else:
            self._authenticate()
            self.auth_time = now_utc()

    def _refresh_hook(
        self, response: requests.Response, *args: Any, **kwargs: Any
    ) -> Optional[requests.Response]:
        # Deprecated: response-hook based refresh is no longer used. Keep for backward
        # compatibility but perform no automatic resend here to avoid recursion during long
        # streaming/bundle operations. Token refresh is handled proactively before requests
        # and on 401 retry paths where applicable.
        try:
            if response.status_code == requests.codes.unauthorized:
                logger.info("Received 401 in response hook; triggering token refresh only.")
                self.refresh_token()
        except Exception:
            logger.error("Token refresh in deprecated response hook failed.")
            logger.error("", exc_info=True)
        # Do not resend the request from the hook; let caller handle retry.
        return response
