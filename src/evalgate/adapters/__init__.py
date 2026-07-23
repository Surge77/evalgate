"""Target adapters: bridge between evalgate cases and the system under eval."""

from evalgate.adapters.base import Adapter
from evalgate.adapters.callable import CallableAdapter
from evalgate.adapters.http import HttpAdapter

__all__ = ["Adapter", "CallableAdapter", "HttpAdapter"]
