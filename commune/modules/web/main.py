# Fix formatting issue by avoiding f-string formatting errors
import os

base = "./lendy_whitepaper"
os.makedirs(base, exist_ok=True)

ascii_header = r"""
\begin{verbatim}
 _      ____  _  _  ____  __   __  ____ 
/ \  /|/  _ \/ \/ \/  _ \/  \ /  \/_   \
| |\ ||| / \|| || || | \||  | ||  | /   /
| | \||| \_/|| || || |_/||  \_/|  |/   /_
\_/  \|\____/\_/\_/\____/\____/\_/\____/
     L E N D Y   P R O T O C O L
\end{verbatim}
"""

main_tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{enumitem}
\usepackage{tikz}
\usepackage{microtype}

\hypersetup{
    colorlinks=true,
    linkcolor=cyan,
    urlcolor=magenta,
    citecolor=cyan
}

\title{\huge\bfseries LENDY:\\Any-to-Any Asset Loan System with Vault Streaming}
\author{}
\date{\today}

\begin{document}
\maketitle

""" + ascii_header + r"""

\section*{Abstract}
Lendy is a decentralized, cyberpunk-inspired lending protocol that allows lenders and borrowers to form mutual contracts for any asset pair across any chain. Using programmable \textbf{vaults}, parties can lock collateral, define vesting schedules, and stream token emissions over time to any address, anywhere.

\section{Lore and Motivation}
In the neon-lit sprawl of decentralized finance, trust is scarce and liquidity is gold. Lendy emerges as a neutral protocol for \textit{street-level credit agreements}: programmable, enforceable, and chain-agnostic.

\section{Core Concepts}
\subsection*{Vaults}
A \textbf{Vault} is a programmable lockbox:
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{Owner Key}: Primary control key (arbitrary-length byte vector).
    \item \textbf{Key Set}: Sub-keys for streaming and asset-specific ops.
    \item \textbf{Streaming Engine}: Emits tokens over time to any target address (vector of bytes) on any chain.
\end{itemize}

\begin{verbatim}
Vault := (owner_key, {sub_keys}, {streams})
Stream := (token_id, amt_per_day, dest_chain, dest_addr_vec, start, end)
\end{verbatim}

\subsection*{Loan Contracts}
\begin{enumerate}[leftmargin=1.5em]
    \item Lender posts loan asset, accepted collateral, vesting schedule.
    \item Borrower posts desired asset, offered collateral, vesting schedule.
    \item Mutual agreement locks collateral + initiates repayment stream.
\end{enumerate}

\section{Cross-Chain Address Model}
All addresses are stored as \texttt{bytes}:
\begin{verbatim}
address_vec ∈ {0,1}*
\end{verbatim}
Supports EVM, Solana, Bitcoin, and future formats without upgrade.

\section{Protocol Flow}
\begin{enumerate}[leftmargin=1.5em]
    \item Offer creation
    \item Discovery
    \item Agreement
    \item Vault lock + stream start
    \item Monitoring + enforcement
    \item Finalization
\end{enumerate}

\section{Streaming Logic}
\begin{lstlisting}[language=Python]
def create_stream(token, amt_per_day, dest_chain, dest_addr_vec, start, end):
    assert vault_balance[token] >= amt_per_day * days
    stream_id = hash(params)
    vault_streams[stream_id] = params
\end{lstlisting}

\section{Security and Enforcement}
\begin{itemize}[leftmargin=1.5em]
    \item Escrow enforcement at vault level
    \item Optional zk/fault proofs for cross-chain verification
    \item Liquidation rules if stream halts
\end{itemize}

\section{Economic Model}
\begin{itemize}[leftmargin=1.5em]
    \item Interest = function of repayment vs principal
    \item Collateralization ratio adjustable
    \item Streaming yields predictable cashflows
\end{itemize}

\section*{Closing Transmission}
Lendy is more than a protocol: it's an \textit{autonomous street lender} in the cyberpunk metaverse, adapting to any chain, any asset, any deal.

\end{document}
"""

with open(os.path.join(base, "main.tex"), "w") as f:
    f.write(main_tex)

with open(os.path.join(base, "README.md"), "w") as f:
    f.write("# Lendy Whitepaper\nCompile with:\n```bash\npdflatex main.tex\n```\nCyberpunk ASCII LaTeX document.\n")
