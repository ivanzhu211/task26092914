"""Multiprocess tests for token_gateway."""

import multiprocessing as mp
import os
import random
import signal
import tempfile
import time
import unittest

from token_gateway import TokenGateway, RefreshError


def _fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # gateway must create it from scratch
    return path


def _concurrent_refresh_worker(db_path, rank, n, barrier, token_queue,
                               result_queue):
    # Every process opens the (possibly brand-new) file at the same time.
    gw = TokenGateway(db_path)
    if rank == 0:
        token = gw.login("user1")
        for _ in range(n - 1):
            token_queue.put(token)
    else:
        token = token_queue.get()
    barrier.wait()
    try:
        result_queue.put(("ok", gw.refresh(token)))
    except Exception as err:  # noqa: BLE001 - report anything to parent
        result_queue.put(("err", repr(err)))


def _refresh_then_report(db_path, token, wfd):
    # Child: rotate, report the successor, then linger until killed.
    gw = TokenGateway(db_path)
    successor = gw.refresh(token)
    os.write(wfd, successor.encode("ascii"))
    time.sleep(120)


def _refresh_once(db_path, token, result_queue):
    gw = TokenGateway(db_path)
    try:
        result_queue.put(("ok", gw.refresh(token)))
    except RefreshError as err:
        result_queue.put(("err", err.reason))


class TestConcurrentOpenAndRefresh(unittest.TestCase):
    def test_simultaneous_first_open_and_refresh(self):
        ctx = mp.get_context("spawn")
        n = 8
        db_path = _fresh_db()
        try:
            barrier = ctx.Barrier(n)
            token_q = ctx.Queue()
            result_q = ctx.Queue()
            procs = [ctx.Process(target=_concurrent_refresh_worker,
                                 args=(db_path, i, n, barrier, token_q,
                                       result_q))
                     for i in range(n)]
            for p in procs:
                p.start()
            for p in procs:
                p.join(60)
                self.assertFalse(p.is_alive(), "worker hung")
                self.assertEqual(p.exitcode, 0)
            results = [result_q.get() for _ in range(n)]
            self.assertTrue(all(s == "ok" for s, _ in results), results)
            successors = {v for _, v in results}
            self.assertEqual(len(successors), 1,
                             f"fork detected: {successors}")
        finally:
            os.unlink(db_path)


class TestSigkillRetry(unittest.TestCase):
    def test_kill_mid_refresh_then_retry(self):
        # fork so the child inherits the report pipe's fd; each process
        # still opens its own fresh SQLite connections after forking.
        ctx = mp.get_context("fork")
        db_path = _fresh_db()
        try:
            gw = TokenGateway(db_path)
            current = gw.login("user2")
            for round_no in range(8):
                rfd, wfd = os.pipe()
                proc = ctx.Process(target=_refresh_then_report,
                                   args=(db_path, current, wfd))
                proc.start()
                os.close(wfd)
                # Kill at a random moment: sometimes before the child even
                # opens the DB, sometimes after it committed and reported.
                time.sleep(random.choice([0.0, 0.005, 0.02,
                                          random.uniform(0, 0.4)]))
                reported = b""
                os.set_blocking(rfd, False)
                try:
                    chunk = os.read(rfd, 4096)
                    if chunk:
                        reported = chunk
                except BlockingIOError:
                    pass
                os.close(rfd)
                os.kill(proc.pid, signal.SIGKILL)
                proc.join(10)

                # Retry the same old token from a different process.
                result_q = ctx.Queue()
                retry = ctx.Process(target=_refresh_once,
                                    args=(db_path, current, result_q))
                retry.start()
                retry.join(30)
                self.assertFalse(retry.is_alive(), "retry hung")
                status, value = result_q.get()
                self.assertEqual(status, "ok",
                                 f"round {round_no}: retry failed: {value}")
                self.assertNotEqual(value, "reuse")
                if reported:
                    self.assertEqual(value.encode("ascii"), reported,
                                     f"round {round_no}: successor mismatch")
                current = value  # chain must keep rotating afterwards
        finally:
            os.unlink(db_path)


class TestReuseFusesSession(unittest.TestCase):
    def test_reuse_then_tail_revoked(self):
        db_path = _fresh_db()
        try:
            gw = TokenGateway(db_path)
            t0 = gw.login("user3")
            t1 = gw.refresh(t0)
            t2 = gw.refresh(t1)  # successor of t0 now used
            with self.assertRaises(RefreshError) as ctx:
                gw.refresh(t0)
            self.assertEqual(ctx.exception.reason, "reuse")
            with self.assertRaises(RefreshError) as ctx:
                gw.refresh(t2)
            self.assertEqual(ctx.exception.reason, "revoked")
            # Revocation is persistent across processes/connections.
            gw2 = TokenGateway(db_path)
            with self.assertRaises(RefreshError) as ctx:
                gw2.refresh(t2)
            self.assertEqual(ctx.exception.reason, "revoked")
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
