"""Async HTTP client using asyncio + aiohttp.

Core async concepts demonstrated here:
- `async def` defines a coroutine (a function that can pause).
- `await` pauses execution until the I/O completes.
- `asyncio.gather()` runs multiple coroutines CONCURRENTLY.
- `asyncio.Semaphore` limits how many run at the same time.
"""

import asyncio
import time
from typing import Sequence

import aiohttp

from neuralcorp.config import settings
from neuralcorp.exceptions import FetchError
from neuralcorp.fetcher.models import FetchResult
from neuralcorp.utils import get_logger

logger = get_logger(__name__)


async def fetch_one(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
) -> FetchResult:
    """Fetch a single URL asynchronously.

    The Semaphore ensures we never open more than
    `max_concurrent_requests` connections at once — polite to servers
    and safe for your RAM.

    Args:
        session: A shared aiohttp session (reuses TCP connections).
        url: The URL to GET.
        semaphore: Limits concurrent requests.

    Returns:
        A FetchResult with status, size, and timing.

    Raises:
        FetchError: If the request fails with a non-2xx status or
                    a network exception.
    """
    start = time.perf_counter()  # High-resolution timer

    # `async with semaphore` decrements the counter; waits if at max
    async with semaphore:
        try:
            logger.debug("Fetching %s", url)

            # `async with session.get(url)` sends the request
            # `await` pauses HERE — event loop can run other coroutines
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=settings.request_timeout_seconds),
            ) as response:
                body = await response.read()  # await: read body bytes
                elapsed = (time.perf_counter() - start) * 1000  # → ms

                result = FetchResult(
                    url=url,
                    status_code=response.status,
                    content_length=len(body),
                    elapsed_ms=round(elapsed, 2),
                    success=200 <= response.status < 300,
                )

                if not result.success:
                    logger.warning("Non-2xx response for %s: %s", url, response.status)
                else:
                    logger.info(
                        "✓ %s → %s (%d bytes, %.1f ms)",
                        url,
                        response.status,
                        result.content_length,
                        elapsed,
                    )

                return result

        except aiohttp.ClientError as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Network error for %s: %s", url, exc)
            # Some error types do not provide an HTTP status code; use 0 to
            # indicate "unknown" instead of None to satisfy type checks.
            raise FetchError(url=url, status_code=0, message=str(exc)) from exc


async def fetch_many(urls: Sequence[str]) -> list[FetchResult]:
    """Fetch multiple URLs concurrently using asyncio.gather().

    This is the key function. Without async, 10 URLs fetched sequentially
    might take 10 × 500 ms = 5 seconds. With gather(), they all run
    concurrently — total time ≈ slowest single request (~500 ms).

    Args:
        urls: A sequence of URLs to fetch.

    Returns:
        A list of FetchResult objects in the same order as `urls`.
        Failed requests return a FetchResult with success=False.
    """
    logger.info("Starting concurrent fetch of %d URLs", len(urls))

    # Semaphore: at most `max_concurrent_requests` active at once
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    # One shared session = efficient TCP connection pooling
    async with aiohttp.ClientSession() as session:
        # Build a list of coroutines — NOT started yet
        tasks = [fetch_one(session, url, semaphore) for url in urls]

        # asyncio.gather() starts ALL coroutines concurrently
        # return_exceptions=True: failed tasks return the exception
        # as a value instead of cancelling everything else
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # `raw_results` has type list[FetchResult | BaseException].
    # Static type checkers can be unhappy when appending a union to a
    # list[FetchResult], so narrow/cast the non-exception branch.
    from typing import cast

    results: list[FetchResult] = []
    for url, result in zip(urls, raw_results):
        if isinstance(result, Exception):
            # Turn exceptions into failed FetchResults — graceful degradation
            logger.warning("Swallowing error for %s: %s", url, result)
            results.append(FetchResult.from_error(url=url, elapsed_ms=0.0))
        else:
            # `result` is not an Exception here; cast for the type checker.
            results.append(cast(FetchResult, result))

    success_count = sum(1 for r in results if r.success)
    logger.info("Fetch complete: %d/%d succeeded", success_count, len(results))
    return results
