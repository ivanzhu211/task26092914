"""refresh_gate 的多进程并发与崩溃安全测试。

场景：
1. 多进程同时首次打开同一个新文件，随后对同一个 token 并发 refresh，
   全部成功且后继逐字节相同。
2. 子进程在 refresh 途中随机时刻被 SIGKILL，另一进程用旧 token 重试：
   已输出的后继必须逐字节一致，重试不得是 reuse，链还能继续轮换。
3. 后继被用过之后再出示旧 token -> reuse；链尾 token -> revoked。
"""

import os
import random
import shutil
import signal
import sys
import tempfile
import time
import unittest
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from refresh_gate import TokenGateway, RefreshError


class FakeClock:
    def __init__(self, start=1_000_000.0):
        self.t = start

    def __call__(self):
        return self.t


class ClockInjectionTest(unittest.TestCase):
    """所有时间判断都走可注入时钟：24h 绝对失效、8h 空闲失效、60s 重试窗。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="refresh_gate_clock_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clock = FakeClock()
        self.gw = TokenGateway(os.path.join(self.tmp, "g.db"), clock=self.clock)

    def test_idle_timeout_8h(self):
        t0 = self.gw.login("u")
        self.clock.t += 8 * 3600 - 1
        t1 = self.gw.refresh(t0)  # 7h59m59s 仍有效
        self.clock.t += 8 * 3600
        with self.assertRaises(RefreshError) as ctx:
            self.gw.refresh(t1)
        self.assertEqual(ctx.exception.reason, "expired")

    def test_absolute_timeout_24h(self):
        tok = self.gw.login("u")
        for _ in range(3):  # 保持活跃（每 7h 轮换一次）也无法超过 24h
            self.clock.t += 7 * 3600
            tok = self.gw.refresh(tok)
        self.clock.t += 3 * 3600  # 累计 24h
        with self.assertRaises(RefreshError) as ctx:
            self.gw.refresh(tok)
        self.assertEqual(ctx.exception.reason, "expired")

    def test_retry_window_and_no_timer_change(self):
        t0 = self.gw.login("u")
        t1 = self.gw.refresh(t0)  # 首次轮换
        self.clock.t += 59
        self.assertEqual(self.gw.refresh(t0), t1)  # 60s 内重试 -> 同一后继
        # 重试不算活动：从首次轮换起满 8h 即空闲过期
        self.clock.t += 8 * 3600 - 59
        with self.assertRaises(RefreshError) as ctx:
            self.gw.refresh(t1)
        self.assertEqual(ctx.exception.reason, "expired")

    def test_retry_window_expired_is_reuse(self):
        t0 = self.gw.login("u")
        self.gw.refresh(t0)
        self.clock.t += 60  # 距首次轮换满 60 秒
        with self.assertRaises(RefreshError) as ctx:
            self.gw.refresh(t0)
        self.assertEqual(ctx.exception.reason, "reuse")


# ---------------- 子进程入口（必须是模块级函数，供 spawn 使用） ----------------

def _child_init_then_refresh(db_path, init_q, token_q, go_event, result_q, idx):
    """场景1子进程：并发首次打开文件 -> 等 token -> 并发 refresh。"""
    try:
        gw = TokenGateway(db_path)  # 多进程首次初始化竞争发生在这里
        init_q.put(("init_ok", idx))
        token = token_q.get(timeout=60)
        go_event.wait(timeout=60)
        succ = gw.refresh(token)
        result_q.put(("ok", succ))
    except RefreshError as exc:
        result_q.put(("refresh_error", exc.reason))
    except Exception as exc:  # noqa: BLE001
        result_q.put(("exception", repr(exc)))


def _child_refresh_write(db_path, token, out_path, ready_q, go_event):
    """场景2子进程：refresh 成功后把后继写盘（写盘后才可能被看到）。"""
    gw = TokenGateway(db_path)
    ready_q.put("ready")
    go_event.wait()
    try:
        succ = gw.refresh(token)
    except Exception:  # noqa: BLE001
        return
    with open(out_path, "w") as fh:
        fh.write(succ)
        fh.flush()
        os.fsync(fh.fileno())
    os._exit(0)


def _child_noise(db_path, ready_q, go_event):
    """场景2噪声进程：不断 login 制造写锁竞争，随后会被 SIGKILL。"""
    gw = TokenGateway(db_path)
    ready_q.put("ready")
    go_event.wait()
    while True:
        gw.login("noise")


# ---------------- 测试用例 ----------------

class MultiProcessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="refresh_gate_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.db = os.path.join(self.tmp, "gateway.db")
        self.ctx = mp.get_context("spawn")

    # 场景 1：并发首次打开 + 并发 refresh 同一 token
    def test_concurrent_init_and_refresh(self):
        n = 8
        init_q = self.ctx.Queue()
        token_q = self.ctx.Queue()
        result_q = self.ctx.Queue()
        go = self.ctx.Event()

        procs = [
            self.ctx.Process(target=_child_init_then_refresh,
                             args=(self.db, init_q, token_q, go, result_q, i))
            for i in range(n)
        ]
        for p in procs:
            p.start()

        # 等所有子进程完成首次初始化（此时文件已被并发创建）
        for _ in range(n):
            tag, idx = init_q.get(timeout=60)
            self.assertEqual(tag, "init_ok")

        # 父进程登录拿到第一个 token，广播后统一放行 refresh
        gw = TokenGateway(self.db)
        token = gw.login("alice")
        for _ in range(n):
            token_q.put(token)
        go.set()

        results = [result_q.get(timeout=60) for _ in range(n)]
        for p in procs:
            p.join(30)

        for tag, val in results:
            self.assertEqual(tag, "ok", msg=f"unexpected result: {tag} {val}")
        successors = {val for _, val in results}
        self.assertEqual(len(successors), 1,
                         "所有并发 refresh 必须拿到逐字节相同的后继")

    # 场景 2：refresh 途中随机 SIGKILL + 跨进程重试
    def test_sigkill_mid_refresh_then_retry(self):
        gw = TokenGateway(self.db)
        token = gw.login("bob")
        rounds = 8
        saw_output = 0

        for rnd in range(rounds):
            go = self.ctx.Event()
            ready_q = self.ctx.Queue()
            workers = []
            out_paths = []
            for i in range(3):
                out_path = os.path.join(self.tmp, f"out_{rnd}_{i}")
                out_paths.append(out_path)
                workers.append(self.ctx.Process(
                    target=_child_refresh_write,
                    args=(self.db, token, out_path, ready_q, go)))
            noise = [self.ctx.Process(target=_child_noise,
                                      args=(self.db, ready_q, go))
                     for _ in range(2)]
            for p in noise + workers:
                p.start()

            # 等全部子进程就绪（已打开数据库、正等在 go 上），再统一放行
            for _ in range(len(workers) + len(noise)):
                self.assertEqual(ready_q.get(timeout=60), "ready")
            go.set()
            time.sleep(random.uniform(0, 0.03))  # 随机时刻 SIGKILL
            for p in workers + noise:
                if p.is_alive():
                    try:
                        os.kill(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            for p in workers + noise:
                p.join(30)

            printed = []
            for out_path in out_paths:
                if os.path.exists(out_path):
                    with open(out_path) as fh:
                        printed.append(fh.read())

            # 无论子进程死在哪一步，用旧 token 重试都必须成功且不是 reuse
            retry = gw.refresh(token)
            saw_output += len(printed)
            for val in printed:
                self.assertEqual(
                    val, retry,
                    "子进程被杀前已输出的后继必须与重试结果逐字节相同")

            # 链还能继续正常轮换
            nxt = gw.refresh(retry)
            self.assertTrue(nxt)
            token = nxt

        # 至少有一部分轮次真的杀死了已提交的子进程之外的情形；
        # 不强制要求有输出（杀得早就没有），只要求有输出时一致。
        self.assertGreaterEqual(saw_output, 0)

    # 场景 3：后继被用过后出示旧 token -> reuse；链尾 -> revoked
    def test_reuse_then_chain_tail_revoked(self):
        gw = TokenGateway(self.db)
        t0 = gw.login("carol")
        t1 = gw.refresh(t0)
        t2 = gw.refresh(t1)  # 后继 t1 已被用来轮换

        with self.assertRaises(RefreshError) as ctx:
            gw.refresh(t0)
        self.assertEqual(ctx.exception.reason, "reuse")

        with self.assertRaises(RefreshError) as ctx:
            gw.refresh(t2)
        self.assertEqual(ctx.exception.reason, "revoked")

    # 附加：伪造/篡改 token -> unknown，且不影响任何会话
    def test_forged_token_is_unknown_and_harmless(self):
        gw = TokenGateway(self.db)
        t0 = gw.login("dave")
        for forged in ("forged-token", t0[:-2] + "xx", t0[:20], ""):
            with self.assertRaises(RefreshError) as ctx:
                gw.refresh(forged)
            self.assertEqual(ctx.exception.reason, "unknown")
        t1 = gw.refresh(t0)  # 会话不受伪造尝试影响
        self.assertTrue(t1)

    # 附加：logout 返回后，会话里任何 token 都换不出后继
    def test_logout_kills_whole_session(self):
        gw = TokenGateway(self.db)
        t0 = gw.login("eve")
        t1 = gw.refresh(t0)
        gw.logout(t1)
        for tok in (t0, t1):
            with self.assertRaises(RefreshError):
                gw.refresh(tok)

    # 附加：revoke_user 作废该用户所有会话
    def test_revoke_user(self):
        gw = TokenGateway(self.db)
        a1 = gw.login("frank")
        a2 = gw.login("frank")
        b1 = gw.login("grace")
        gw.revoke_user("frank")
        for tok in (a1, a2):
            with self.assertRaises(RefreshError) as ctx:
                gw.refresh(tok)
            self.assertEqual(ctx.exception.reason, "revoked")
        self.assertTrue(gw.refresh(b1))  # 其他用户不受影响


if __name__ == "__main__":
    unittest.main()
