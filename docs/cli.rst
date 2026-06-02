CLI reference
=============

Installing the package registers the ``ophir`` console script (entry point
``ophir.cli:app``), a `Typer <https://typer.tiangolo.com/>`_ application. The
commands below are documented manually because Typer apps are not introspectable
by the standard Sphinx CLI extensions; the underlying functions are also
available in the :doc:`API reference <api/index>` (:mod:`ophir.cli`).

``ophir serve``
---------------

Launch the Ophir Gradio UI (:func:`ophir.ui.serve`).

.. code-block:: bash

   ophir serve [--port INTEGER] [--share / --no-share] [--debug / --no-debug]

============== ========= ============================================
Option         Default   Description
============== ========= ============================================
``--port``     ``7860``  Gradio server port.
``--share``    ``False`` Expose a public Gradio share link.
``--debug``    ``True``  Launch Gradio in debug mode.
============== ========= ============================================

.. note::

   ``serve`` imports and launches :mod:`ophir.ui`, which at import time fetches
   the S&P 500 list and split history from the network and loads a trained base
   checkpoint onto a CUDA device. A GPU, a checkpoint, network access, and a
   local Ollama server are therefore required for this command to run.

``ophir register massive-key``
------------------------------

Store a `MASSIVE <https://pypi.org/project/massive/>`_ API key for data
fetching. The key is written to ``.massive_key`` inside the package's
``.ophir/`` directory.

.. code-block:: bash

   ophir register massive-key <KEY>

============== ============================================
Argument       Description
============== ============================================
``KEY``        The MASSIVE API key to store.
============== ============================================

The stored key is later read by :func:`ophir.register.get_massive_client` to
construct an authenticated ``massive.RESTClient``.

``ophir trade``
---------------

Alpaca paper-trading commands (:mod:`ophir.trading`). Credentials are stored as
JSON under the package's ``.ophir/alpaca_paper.json``; all orders go to Alpaca's
paper endpoint.

.. code-block:: bash

   ophir trade configure                       # save paper API key + secret
   ophir trade account                         # cash / portfolio / buying power
   ophir trade buy SYMBOL [--qty N | --notional $]
   ophir trade sell SYMBOL [--qty N]
   ophir trade orders                          # list open orders
   ophir trade positions                       # list positions with P&L

``ophir trade rebalance``
-------------------------

Score the S&P 500 with the trained model and rebalance the paper account into
the top-ranked names (:func:`ophir.strategy.score_universe`,
:func:`ophir.strategy.rank_top_k`, :func:`ophir.strategy.compute_rebalance`).
Defaults to a **dry run** that prints the plan without trading.

.. code-block:: bash

   ophir trade rebalance [--top-k INTEGER] [--horizon INTEGER]
                         [--budget-frac FLOAT] [--min-score FLOAT]
                         [--refresh / --no-refresh] [--dry-run / --execute]

================= ============= ==================================================
Option            Default       Description
================= ============= ==================================================
``--top-k``       ``10``        Number of highest-scoring symbols to hold.
``--horizon``     ``5``         Forecast days summed into each symbol's score.
``--budget-frac`` ``0.95``      Fraction of buying power deployed across new buys.
``--min-score``   ``0.0``       Minimum predicted return required to buy.
``--refresh``     ``False``     Refresh OHLC data from MASSIVE before scoring.
``--dry-run``     ``True``      Print the plan; pass ``--execute`` to place orders.
================= ============= ==================================================

.. note::

   ``rebalance`` loads a trained base checkpoint onto a CUDA device and runs
   model inference over the S&P 500, so a GPU, a checkpoint, and (with
   ``--refresh``) a MASSIVE API key are required. It places orders only on the
   Alpaca paper account, and only when ``--execute`` is given.
