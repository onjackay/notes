# tmux: A Comprehensive Guide

tmux (**T**erminal **Mu**ltiple**x**er) lets you run multiple terminal sessions inside a single terminal window. It persists even if your terminal closes, so you can detach and reattach later — great for remote work and long-running processes.

---

## 1. Core Concepts

tmux has three layers:

| Layer | What it is | Analogy |
|---|---|---|
| **Session** | A named collection of windows | A "workspace" or "project" |
| **Window** | A full-screen tab within a session | A browser tab |
| **Pane** | A split within a window | Split panes in Vim/VS Code |

```
Session: "myproject"
├── Window 0: "editor"       (vim)
├── Window 1: "server"       (npm run dev)
│   ├── Pane 0: server logs
│   └── Pane 1: git status (split vertically)
└── Window 2: "shell"
```

---

## 2. Installation

```bash
# macOS
brew install tmux

# Ubuntu/Debian
sudo apt install tmux

# Verify
tmux -V
```

---

## 3. Starting Out

### First session

```bash
tmux                              # Start a new unnamed session
tmux new -s mysession             # Start a named session
```

Once inside, you'll see a green **status bar** at the bottom showing session name, window list, clock, etc.

### Detach & Reattach (the killer feature)

```bash
# Inside tmux, detach:
Ctrl+b  d

# Back in your regular terminal:
tmux ls                           # List sessions
tmux attach -t mysession          # Reattach to named session
tmux attach                       # Reattach to most recent
```

Your programs keep running while detached! Close your terminal, SSH in again, `tmux attach` — everything is still there.

---

## 4. The Prefix Key

All tmux commands start with the **prefix** followed by a key. The default prefix is `Ctrl+b`.

So when I write `Prefix + c`, I mean: press `Ctrl+b`, release both, then press `c`.

> **Tip:** Many people remap prefix to `Ctrl+a` (closer to home row). Add to `~/.tmux.conf`:
> ```
> set -g prefix C-a
> unbind C-b
> bind C-a send-prefix
> ```

---

## 5. Essential Keybindings

### Sessions

| Keys | Action |
|---|---|
| `Prefix + d` | Detach from session |
| `Prefix + $` | Rename current session |
| `Prefix + s` | List sessions (interactive) |
| `Prefix + (` | Switch to previous session |

### Windows (tabs)

| Keys | Action |
|---|---|
| `Prefix + c` | Create new window |
| `Prefix + ,` | Rename current window |
| `Prefix + w` | List windows (interactive preview) |
| `Prefix + n` | Next window |
| `Prefix + p` | Previous window |
| `Prefix + 0-9` | Jump to window by number |
| `Prefix + &` | Kill current window (confirm with `y`) |

### Panes (splits)

| Keys | Action |
|---|---|
| `Prefix + %` | Split vertically (left/right) |
| `Prefix + "` | Split horizontally (top/bottom) |
| `Prefix + o` | Cycle to next pane |
| `Prefix + ;` | Toggle to last active pane |
| `Prefix + arrow` | Move to pane in direction |
| `Prefix + x` | Kill current pane (confirm with `y`) |
| `Prefix + z` | Zoom/unzoom current pane |
| `Prefix + !` | Break pane into its own window |
| `Prefix + space` | Cycle through pane layouts |
| `Prefix + {` / `}` | Swap pane left/right or up/down |

### Copy Mode (scrolling)

| Keys | Action |
|---|---|
| `Prefix + [` | Enter copy mode (scroll with arrows/PageUp/PageDown) |
| `q` | Exit copy mode |
| `Space` | Start selection (in copy mode) |
| `Enter` | Copy selection (in copy mode) |
| `Prefix + ]` | Paste copied text |

> **Better scrolling:** If your tmux is modern enough, enable vi keys and mouse:
> ```
> set -g mode-keys vi
> set -g mouse on
> ```

---

## 6. A Real-World Workflow

```bash
# Start a project session
tmux new -s myapp

# You're now in window 0. Rename it:
Prefix + ,    →  type "editor", Enter

# Start your editor
vim src/app.ts

# Create a window for the dev server:
Prefix + c
Prefix + ,    →  type "server", Enter
npm run dev

# Create a window for git/shell:
Prefix + c
Prefix + ,    →  type "shell", Enter

# Detach and go about your day:
Prefix + d

# Later, reattach:
tmux attach -t myapp
```

### Splitting within one window

```
Prefix + %     → vertical split (side-by-side)
Prefix + "     → horizontal split (stacked)
Prefix + z     → zoom one pane to full screen
Prefix + z     → unzoom back
```

---

## 7. Useful Config (`~/.tmux.conf`)

```bash
# Better prefix (optional)
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# Vi-style copy mode
set -g mode-keys vi

# Enable mouse (click panes, resize, scroll)
set -g mouse on

# Start window numbering at 1 (easier keyboard reach)
set -g base-index 1
setw -g pane-base-index 1

# Faster escape key delay
set -sg escape-time 0

# Increase scrollback buffer
set -g history-limit 50000

# Reload config without restarting
bind r source-file ~/.tmux.conf \; display "Config reloaded!"
```

Apply changes: `tmux source-file ~/.tmux.conf` or `Prefix + r` (if you added the reload bind).

---

## 8. Command-Line Power

You can script tmux from outside:

```bash
# Create session with a command running
tmux new -s watch -d 'htop'

# Send keys to a running session
tmux send-keys -t mysession:editor 'gg' C-m

# Capture pane content
tmux capture-pane -t mysession:server -p

# Kill a session
tmux kill-session -t mysession
```

---

## 9. Quick Reference Card

Here's a minimal cheat sheet. Practice these daily:

```
Prefix + c       New window
Prefix + n/p     Next/previous window
Prefix + 1-9     Jump to window
Prefix + %       Vertical split
Prefix + "       Horizontal split
Prefix + arrows  Move between panes
Prefix + d       Detach
Prefix + [       Enter scroll/copy mode
Prefix + ]       Paste
```

---

## 10. Next Steps

1. **Start small:** Use tmux on your next SSH session
2. **Practice daily:** 5 keys at a time until they become muscle memory
3. **Explore:** Try `tmux list-keys` to see all keybindings
4. **Customize:** Build a `~/.tmux.conf` that fits your workflow
5. **Level up:** Look into tmux plugins with [TPM](https://github.com/tmux-plugins/tpm) (session saving, theming, etc.)

---

## 11. Hands-On Practice: Build a Tiny Dashboard

A 5-minute exercise covering the most important tmux moves.

### Step 1: Create a session

```bash
tmux new -s practice
```

You're now inside tmux (notice the green status bar at the bottom).

### Step 2: Make a vertical split

Press: `Prefix + %` — that's `Ctrl+b`, release, then `Shift+5`

You now have **two panes** side by side. Cursor is in the right pane.

### Step 3: Run a clock in the right pane

Type:

```bash
watch -n 1 date
```

(Updates every second. Press `Ctrl+c` anytime to kill it later.)

### Step 4: Jump back to the left pane

Press: `Prefix + ←` (left arrow)

### Step 5: Split the left pane horizontally

Press: `Prefix + "` — that's `Ctrl+b`, release, then `"`

Now you have **three panes**: top-left, bottom-left, and right.

### Step 6: Fill the bottom-left pane

Type:

```bash
ls -la
```

### Step 7: Jump between all three panes

Use `Prefix + arrow keys` to navigate. Notice the highlight border follows you.

### Step 8: Zoom one pane

Navigate to the clock pane, then: `Prefix + z`

The clock fills the whole window. Press `Prefix + z` again to unzoom.

### Step 9: Create a second window

Press: `Prefix + c`

This is a fresh empty screen. Run anything:

```bash
top
```

### Step 10: Switch between windows

- `Prefix + 0` — back to window 0 (your 3-pane dashboard)
- `Prefix + 1` — back to window 1 (top)

### Step 11: The big finale — detach and reattach

Press: `Prefix + d`

You're back in plain terminal. Everything is still running.

```bash
tmux ls              # see your "practice" session
tmux attach          # reattach — everything exactly as you left it
```

### Step 12: Clean up

Inside the session, type `exit` in each pane (or `Prefix + x` then `y` to kill panes). When the last pane exits, the window dies. When the last window dies, the session dies.

Or just:

```bash
tmux kill-session -t practice
```

### What you just practiced

| You learned | Key |
|---|---|
| Start a named session | `tmux new -s practice` |
| Vertical split | `Prefix + %` |
| Horizontal split | `Prefix + "` |
| Move between panes | `Prefix + arrows` |
| Zoom a pane | `Prefix + z` |
| New window | `Prefix + c` |
| Switch windows | `Prefix + 0`, `Prefix + 1` |
| Detach | `Prefix + d` |
| Reattach | `tmux attach` |

That's the 80/20. Do this loop 3 times and it'll stick.
