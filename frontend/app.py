from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.dropdown import DropDown
from kivy.uix.checkbox import CheckBox
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import asyncio
from threading import Thread
import time
import json
import cProfile
import pstats
import torch
from config import MAX_PORTFOLIO_RISK, TRADING_PAIRS, DEFAULT_USERNAME, DEFAULT_PASSWORD, GAN_TUNING_PAIR, MAX_TRADE_SIZE
from backend.login import LoginManager
from backend.backtester import Backtester
from backend.alerts import AlertsManager
import logging
import os
import cryptography.fernet
from cryptography.fernet import Fernet

# Structured JSON logging
logging.basicConfig(
    filename='logs/bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/bot.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# Encryption key for state.json
key = os.getenv("ENCRYPTION_KEY") or Fernet.generate_key()
cipher = Fernet(key)

class TradingApp(App):
    def __init__(self, api, model, risk_manager, strategy_manager, simulate=True):
        """Initialize TradingApp with all components.

        Args:
            api (KrakenAPI): Kraken API instance.
            model (EnsembleModel): Ensemble model instance with api passed.
            risk_manager (RiskManager): Risk management instance.
            strategy_manager (StrategyManager): Strategy management instance.
            simulate (bool): Whether to run in simulation mode.
        """
        super().__init__()
        self.api = api
        self.model = model  # Use the passed model instance with api
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        self.alerts_manager = AlertsManager()
        self.backtester = Backtester(api, self.model, strategy_manager)
        self.simulate = simulate
        self.login_mgr = LoginManager()
        self.running = False
        self.logged_in = False
        self.username = None
        self.use_gpu = torch.cuda.is_available()
        logger.info({"event": "init", "gpu_available": self.use_gpu})
        self.latest_data = {pair: {} for pair in TRADING_PAIRS}
        self.prices = {pair: [] for pair in TRADING_PAIRS}
        self.predictions = {pair: {
            'Ensemble': [], 'LSTM': [], 'GRU': [], 'Transformer': [], 'TCN': [], 
            'PPO': [], 'ActorCritic': [], 'GNN': [], 'TransferLearning': [], 'RandomForest': []
        } for pair in TRADING_PAIRS}
        self.risk_levels = {pair: [] for pair in TRADING_PAIRS}
        self.profits = []
        self.sentiment_scores = {pair: [] for pair in TRADING_PAIRS}
        self.trades = {'wins': 0, 'losses': 0}
        self.initial_balance = self.load_initial_balance()
        self.figures = {pair: plt.subplots(figsize=(6, 4)) for pair in TRADING_PAIRS}
        for pair, (fig, ax) in self.figures.items():
            ax.set_title(f"{pair} Price & Predictions")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (USD)")
            self.figures[pair] = (fig, ax, ax.twinx())
            self.figures[pair][2].set_ylabel("Portfolio Risk/Sentiment (%)")
            self.figures[pair][2].set_ylim(0, MAX_PORTFOLIO_RISK * 200)

    def load_initial_balance(self):
        """Load initial balance from encrypted state.json."""
        try:
            with open('state.json', 'rb') as f:
                encrypted_data = f.read()
                decrypted_data = cipher.decrypt(encrypted_data)
                state = json.loads(decrypted_data.decode())
                self.trades = state['trades']
                balance = state['initial_balance']
                logger.info({"event": "load_state", "status": "success"})
                return balance
        except FileNotFoundError:
            logger.info({"event": "load_state", "status": "no_file", "using_api_balance": True})
            return self.api.get_balance()['USD']
        except Exception as e:
            logger.error({"event": "load_state_error", "error": str(e)})
            return self.api.get_balance()['USD']

    def build(self):
        """Build UI with login screen initially."""
        self.layout = BoxLayout(orientation='vertical')
        if not self.logged_in:
            self.show_login_screen()
        else:
            self.show_trading_screen()
        return self.layout

    def show_login_screen(self):
        """Display login screen."""
        self.layout.clear_widgets()
        self.username_input = TextInput(hint_text="Username", multiline=False, text=DEFAULT_USERNAME)
        self.password_input = TextInput(hint_text="Password", password=True, multiline=False, text=DEFAULT_PASSWORD)
        self.two_factor_input = TextInput(hint_text="2FA Code", multiline=False)
        self.login_button = Button(text="Login", on_press=self.attempt_login)
        self.register_button = Button(text="Register", on_press=self.attempt_register)
        self.status_label = Label(text="Enter credentials to start AiKrakBot")
        
        self.login_mgr.register_user(DEFAULT_USERNAME, DEFAULT_PASSWORD, "1234567890", "test@example.com")
        
        self.layout.add_widget(Label(text="AiKrakBot Login"))
        self.layout.add_widget(self.username_input)
        self.layout.add_widget(self.password_input)
        self.layout.add_widget(self.two_factor_input)
        self.layout.add_widget(self.login_button)
        self.layout.add_widget(self.register_button)
        self.layout.add_widget(self.status_label)

    def show_trading_screen(self):
        """Display trading dashboard after login."""
        self.layout.clear_widgets()
        self.tabs = TabbedPanel(do_default_tab=False)
        
        live_tab = TabbedPanelItem(text="Live Trading")
        live_layout = BoxLayout(orientation='vertical')
        self.price_labels = {pair: Label(text=f"Price ({pair}): $0") for pair in TRADING_PAIRS}
        self.balance_label = Label(text="Balance: $0 / 0 BTC")
        self.pred_labels = {pair: Label(text=f"Predicted Price ({pair}): N/A") for pair in TRADING_PAIRS}
        self.action_labels = {pair: Label(text=f"Action ({pair}): N/A") for pair in TRADING_PAIRS}
        self.risk_label = Label(text="Portfolio Risk: 0%")
        self.stress_label = Label(text="Stress Test Loss: $0")
        self.profit_label = Label(text="Profit: $0")
        self.win_rate_label = Label(text="Win Rate: 0%")
        self.sentiment_labels = {pair: Label(text=f"News Sentiment ({pair}): 0.00") for pair in TRADING_PAIRS}
        self.latency_label = Label(text="Prediction Latency: 0.00s")
        self.fid_label = Label(text="GAN FID: N/A")
        self.mse_moments_label = Label(text="GAN MSE Moments: N/A")
        self.acf_error_label = Label(text="GAN ACF Error: N/A")
        self.stability_label = Label(text="GAN Stability: N/A")
        
        self.strategy_dropdown = DropDown()
        for strat in self.strategy_manager.strategies.keys():
            btn = Button(text=strat, size_hint_y=None, height=30)
            btn.bind(on_release=lambda btn: self.strategy_dropdown.select(btn.text))
            self.strategy_dropdown.add_widget(btn)
        self.strategy_btn = Button(text=f"Strategy: {self.strategy_manager.current_strategy}")
        self.strategy_btn.bind(on_release=self.strategy_dropdown.open)
        self.strategy_dropdown.bind(on_select=lambda instance, x: setattr(self.strategy_manager, 'current_strategy', x) or setattr(self.strategy_btn, 'text', f"Strategy: {x}"))
        
        self.plot_options = {pair: {} for pair in TRADING_PAIRS}
        plot_layout = BoxLayout(orientation='horizontal', size_hint_y=0.2)
        for model in self.predictions[TRADING_PAIRS[0]].keys():
            cb = CheckBox(active=True if model == 'Ensemble' else False)
            self.plot_options[TRADING_PAIRS[0]][model] = cb
            box = BoxLayout(orientation='horizontal')
            box.add_widget(Label(text=model))
            box.add_widget(cb)
            plot_layout.add_widget(box)
        
        self.canvases = {pair: FigureCanvasKivyAgg(fig) for pair, (fig, _, _) in self.figures.items()}
        for pair in TRADING_PAIRS:
            live_layout.add_widget(self.price_labels[pair])
            live_layout.add_widget(self.pred_labels[pair])
            live_layout.add_widget(self.action_labels[pair])
            live_layout.add_widget(self.sentiment_labels[pair])
        live_layout.add_widget(self.balance_label)
        live_layout.add_widget(self.risk_label)
        live_layout.add_widget(self.stress_label)
        live_layout.add_widget(self.profit_label)
        live_layout.add_widget(self.win_rate_label)
        live_layout.add_widget(self.latency_label)
        live_layout.add_widget(self.fid_label)
        live_layout.add_widget(self.mse_moments_label)
        live_layout.add_widget(self.acf_error_label)
        live_layout.add_widget(self.stability_label)
        live_layout.add_widget(self.canvases[TRADING_PAIRS[0]])
        live_layout.add_widget(self.strategy_btn)
        live_layout.add_widget(plot_layout)
        
        settings_btn = Button(text="Settings", on_press=self.show_settings_popup)
        live_layout.add_widget(settings_btn)
        
        self.start_btn = Button(text="Start", on_press=self.start_bot)
        self.stop_btn = Button(text="Stop", on_press=self.stop_bot)
        self.log_view = ScrollView(size_hint=(1, 0.2))
        self.log_display = TextInput(text="", readonly=True, size_hint_y=None, height=100)
        self.log_view.add_widget(self.log_display)
        live_layout.add_widget(self.log_view)
        live_layout.add_widget(self.start_btn)
        live_layout.add_widget(self.stop_btn)
        live_tab.add_widget(live_layout)
        self.tabs.add_widget(live_tab)
        
        backtest_tab = TabbedPanelItem(text="Backtest")
        backtest_layout = BoxLayout(orientation='vertical')
        self.backtest_result = Label(text="Backtest Results: Not run")
        backtest_btn = Button(text="Run Backtest", on_press=self.run_backtest)
        backtest_layout.add_widget(self.backtest_result)
        backtest_layout.add_widget(backtest_btn)
        backtest_tab.add_widget(backtest_layout)
        self.tabs.add_widget(backtest_tab)
        
        self.layout.add_widget(self.tabs)

    def show_settings_popup(self, instance):
        """Show settings popup for live config updates."""
        content = BoxLayout(orientation='vertical')
        self.max_trade_input = TextInput(hint_text=f"MAX_TRADE_SIZE (current: {MAX_TRADE_SIZE})", multiline=False)
        save_btn = Button(text="Save", on_press=self.save_settings)
        content.add_widget(Label(text="Settings"))
        content.add_widget(self.max_trade_input)
        content.add_widget(save_btn)
        self.settings_popup = Popup(title="AiKrakBot Settings", content=content, size_hint=(0.5, 0.5))
        self.settings_popup.open()

    def save_settings(self, instance):
        """Save updated config to .env."""
        try:
            max_trade_size = float(self.max_trade_input.text)
            with open('.env', 'r') as f:
                lines = f.readlines()
            with open('.env', 'w') as f:
                found = False
                for line in lines:
                    if line.startswith("MAX_TRADE_SIZE="):
                        f.write(f"MAX_TRADE_SIZE={max_trade_size}\n")
                        found = True
                    else:
                        f.write(line)
                if not found:
                    f.write(f"MAX_TRADE_SIZE={max_trade_size}\n")
            from config import MAX_TRADE_SIZE as old_max
            import config
            config.MAX_TRADE_SIZE = max_trade_size
            logger.info({"event": "settings_update", "setting": "MAX_TRADE_SIZE", "old_value": old_max, "new_value": max_trade_size})
            self.settings_popup.dismiss()
        except ValueError:
            logger.error({"event": "settings_error", "error": "Invalid MAX_TRADE_SIZE value"})
            self.settings_popup.content.add_widget(Label(text="Error: Invalid number"))

    def run_backtest(self, instance):
        """Run backtest and display results."""
        try:
            profit = self.backtester.run(GAN_TUNING_PAIR, use_synthetic=False)
            self.backtest_result.text = f"Backtest Results: Profit = ${profit:.2f}"
            logger.info({"event": "backtest_complete", "pair": GAN_TUNING_PAIR, "profit": profit})
        except Exception as e:
            logger.error({"event": "backtest_error", "error": str(e)})
            self.backtest_result.text = f"Backtest Failed: {str(e)}"

    def attempt_login(self, instance):
        """Handle login attempt and proceed with GAN tuning and auto-start."""
        username = self.username_input.text
        password = self.password_input.text
        two_factor_code = self.two_factor_input.text
        try:
            token = self.login_mgr.login(username, password, two_factor_code, two_factor_method='sms')
            if token and self.login_mgr.verify_token(token):
                logger.info({"event": "login_success", "username": username})
                self.username = username
                self.load_user_profile()
                self.logged_in = True
                self.tune_gan()
                self.show_trading_screen()
                self.start_bot(None)
            else:
                self.status_label.text = "Login failed. Check credentials and 2FA code (see logs for 2FA)."
                logger.warning({"event": "login_failed", "username": username, "reason": "invalid_credentials"})
        except Exception as e:
            logger.error({"event": "login_error", "error": str(e)})
            self.status_label.text = f"Login error: {str(e)}"

    def attempt_register(self, instance):
        """Handle registration attempt."""
        username = self.username_input.text
        password = self.password_input.text
        try:
            if self.login_mgr.register_user(username, password):
                self.status_label.text = "Registration successful. Please login."
                logger.info({"event": "register_success", "username": username})
            else:
                self.status_label.text = "Registration failed. Username may already exist."
                logger.warning({"event": "register_failed", "username": username, "reason": "username_exists"})
        except Exception as e:
            logger.error({"event": "register_error", "error": str(e)})
            self.status_label.text = f"Registration error: {str(e)}"

    def tune_gan(self):
        """Tune the GAN model with historical data."""
        self.status_label.text = "Tuning GAN, please wait..."
        self.layout.canvas.ask_update()
        logger.info({"event": "gan_tuning_start"})
        try:
            historical_data = self.api.fetch_historical_data(GAN_TUNING_PAIR, limit=1000)
            gan_data = [[d[4], d[5]] for d in historical_data]
            gan_data = [gan_data[i:i+20] for i in range(len(gan_data) - 20)]
            if gan_data:
                logger.info({"event": "gan_tuning_data_fetched", "pair": GAN_TUNING_PAIR, "data_length": len(gan_data)})
                self.model.tune_gan(gan_data)
                logger.info({"event": "gan_tuning_complete"})
                self.status_label.text = "GAN tuning completed"
            else:
                logger.warning({"event": "gan_tuning_failed", "reason": "no_data"})
                self.status_label.text = "GAN tuning failed: No data"
        except Exception as e:
            logger.error({"event": "gan_tuning_error", "error": str(e)})
            self.status_label.text = f"GAN tuning failed: {str(e)}"
        self.layout.canvas.ask_update()

    def load_user_profile(self):
        """Load user-specific preferences."""
        profile_file = f"profiles/{self.username}.json"
        try:
            with open(profile_file, 'r') as f:
                profile = json.load(f)
                for pair in TRADING_PAIRS:
                    if pair in profile['plot_options']:
                        for model, active in profile['plot_options'][pair].items():
                            if model in self.plot_options[pair]:
                                self.plot_options[pair][model].active = active
                logger.info({"event": "load_profile", "username": self.username, "status": "success"})
        except FileNotFoundError:
            logger.info({"event": "load_profile", "username": self.username, "status": "no_file"})
        except Exception as e:
            logger.error({"event": "load_profile_error", "username": self.username, "error": str(e)})

    def save_user_profile(self):
        """Save user-specific preferences."""
        profile_file = f"profiles/{self.username}.json"
        os.makedirs("profiles", exist_ok=True)
        profile = {
            'plot_options': {pair: {model: cb.active for model, cb in options.items()} 
                            for pair, options in self.plot_options.items()}
        }
        try:
            with open(profile_file, 'w') as f:
                json.dump(profile, f)
            logger.info({"event": "save_profile", "username": self.username, "status": "success"})
        except Exception as e:
            logger.error({"event": "save_profile_error", "username": self.username, "error": str(e)})

    def start_bot(self, instance):
        """Start the bot and UI updates."""
        if not self.logged_in:
            logger.warning({"event": "start_bot_failed", "reason": "not_logged_in"})
            return
        self.running = True
        Thread(target=self.run_backend, daemon=True).start()
        Clock.schedule_interval(self.update_ui_async_wrapper, 1)
        logger.info({"event": "bot_started"})

    def stop_bot(self, instance):
        """Stop the bot, save state and profile, and stop UI updates."""
        self.running = False
        Clock.unschedule(self.update_ui_async_wrapper)
        state = {'trades': self.trades, 'initial_balance': self.initial_balance}
        try:
            encrypted_state = cipher.encrypt(json.dumps(state).encode())
            with open('state.json', 'wb') as f:
                f.write(encrypted_state)
            self.save_user_profile()
            logger.info({"event": "bot_stopped", "state_saved": True})
        except Exception as e:
            logger.error({"event": "stop_bot_error", "error": str(e)})
        finally:
            asyncio.get_event_loop().stop()

    def run_backend(self):
        """Run the backend in a separate thread."""
        asyncio.run(self.backend_loop())

    async def backend_loop(self):
        """Process market data asynchronously with reconnection logic."""
        async def on_market_data(pair, price, volume):
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                balance = self.api.get_balance()
                position = self.strategy_manager.strategies['momentum'].positions.get(pair, 0)
                self.model.update_data(price, volume, balance, position, pair)
                self.model.train()
                start_time = time.time()
                prediction = self.model.predict(balance, position)
                latency = time.time() - start_time
                _, portfolio_risk = self.risk_manager.monitor_portfolio_risk(balance)
                stress_loss = sum(self.risk_manager.stress_test(p, self.risk_manager.price_history[p][-1]) 
                                 for p in TRADING_PAIRS if self.risk_manager.price_history[p])
                profit = balance['USD'] - self.initial_balance
                if 'last_trade' in self.latest_data[pair] and self.latest_data[pair]['last_trade']['side'] == 'sell':
                    trade_profit = (price - self.latest_data[pair]['last_trade']['price']) * self.latest_data[pair]['last_trade']['amount']
                    if trade_profit > 0:
                        self.trades['wins'] += 1
                    else:
                        self.trades['losses'] += 1

                if portfolio_risk > MAX_PORTFOLIO_RISK:
                    self.alerts_manager.trigger_alert(f"High portfolio risk: {portfolio_risk*100:.2f}%", self.show_alert_popup)

                self.predictions[pair]['Ensemble'].append(prediction['price'] if prediction and prediction['price'] is not None else price)
                if prediction and 'individual_preds' in prediction:
                    for model_name, pred in prediction['individual_preds'].items():
                        self.predictions[pair][model_name].append(pred if pred is not None else price)

                self.latest_data[pair] = {
                    'pair': pair,
                    'price': price,
                    'balance': balance,
                    'prediction': prediction['price'] if prediction else None,
                    'action': ['Buy', 'Sell', 'Hold'][prediction['action']] if prediction else 'N/A',
                    'individual_preds': prediction['individual_preds'] if prediction else {},
                    'portfolio_risk': portfolio_risk,
                    'stress_loss': stress_loss,
                    'profit': profit,
                    'sentiment_score': prediction['sentiment_score'],
                    'latency': latency,
                    'last_trade': {'price': price, 'amount': position, 'side': 'sell' if prediction['action'] == 1 else 'buy'} 
                                  if prediction['action'] in [0, 1] else self.latest_data[pair].get('last_trade', {})
                }
                self.prices[pair].append(price)
                self.risk_levels[pair].append(portfolio_risk * 100)
                self.profits.append(profit)
                self.sentiment_scores[pair].append(prediction['sentiment_score'] * 100)
                if len(self.prices[pair]) > 50:
                    self.prices[pair].pop(0)
                    self.risk_levels[pair].pop(0)
                    self.profits.pop(0)
                    self.sentiment_scores[pair].pop(0)
                    for preds in self.predictions[pair].values():
                        preds.pop(0)

                self.strategy_manager.update(pair, price)
                self.strategy_manager.execute(pair, price, prediction['price'] if prediction else price, prediction['action'] if prediction else 2)
            except Exception as e:
                logger.error({"event": "market_data_error", "pair": pair, "error": str(e)})
                self.latest_data[pair]['error'] = str(e)
            profiler.disable()
            with open('logs/profile.log', 'a') as f:
                ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
                ps.print_stats()
                logger.info({"event": "profiling_complete", "pair": pair})

        retries = 0
        max_retries = 5
        while retries < max_retries and self.running:
            try:
                await self.api.stream_market_data(on_market_data)
            except Exception as e:
                retries += 1
                delay = 2 ** retries
                logger.error({"event": "websocket_error", "attempt": retries, "max_retries": max_retries, "error": str(e), "retry_delay": delay})
                await asyncio.sleep(delay)
        if retries >= max_retries:
            logger.error({"event": "websocket_failed", "reason": "max_retries_exceeded"})
            self.running = False

    def show_alert_popup(self, message):
        """Show a popup for alerts."""
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text=message))
        close_btn = Button(text="Close", on_press=lambda x: self.alert_popup.dismiss())
        content.add_widget(close_btn)
        self.alert_popup = Popup(title="Alert", content=content, size_hint=(0.5, 0.5))
        self.alert_popup.open()

    async def update_ui_async(self):
        """Asynchronously update the trading UI with latest data and logs."""
        await asyncio.sleep(0.1)  # Small delay to yield control
        for pair in TRADING_PAIRS:
            if self.latest_data[pair]:
                data = self.latest_data[pair]
                self.price_labels[pair].text = f"Price ({pair}): ${data['price']:.2f}"
                self.pred_labels[pair].text = f"Predicted Price ({pair}): ${data['prediction']:.2f}" if data['prediction'] else f"Predicted Price ({pair}): N/A"
                self.action_labels[pair].text = f"Action ({pair}): {data['action']}"
                self.sentiment_labels[pair].text = f"News Sentiment ({pair}): {data['sentiment_score']:.2f}"
                self.balance_label.text = f"Balance: ${data['balance']['USD']:.2f} / {data['balance']['XBT']:.4f} BTC"
                self.risk_label.text = f"Portfolio Risk: {data['portfolio_risk']*100:.2f}%"
                self.stress_label.text = f"Stress Test Loss: ${-data['stress_loss']:.2f}" if data['stress_loss'] < 0 else "Stress Test Loss: $0"
                self.profit_label.text = f"Profit: ${data['profit']:.2f}"
                total_trades = self.trades['wins'] + self.trades['losses']
                win_rate = (self.trades['wins'] / total_trades * 100) if total_trades > 0 else 0
                self.win_rate_label.text = f"Win Rate: {win_rate:.2f}%"
                self.latency_label.text = f"Prediction Latency: {data['latency']:.2f}s"
                metrics = self.model.get_gan_metrics()
                self.fid_label.text = f"GAN FID: {metrics['fid']:.2f}"
                self.mse_moments_label.text = f"GAN MSE Moments: {metrics['mse_moments']:.4f}"
                self.acf_error_label.text = f"GAN ACF Error: {metrics['acf_error']:.4f}"
                self.stability_label.text = f"GAN Stability: {metrics['stability']:.4f}"

                with open('logs/bot.log', 'r') as f:
                    log_text = f.read()[-1000:]
                    if 'error' in data:
                        log_text += f"\nError ({pair}): {data['error']}"
                    self.log_display.text = log_text

                fig, ax, ax2 = self.figures[pair]
                ax.clear()
                ax2.clear()
                ax.plot(self.prices[pair], label='Price', color='blue')
                colors = {
                    'Ensemble': 'orange', 'LSTM': 'lightblue', 'GRU': 'lightgreen', 'Transformer': 'lightcoral',
                    'TCN': 'lightpink', 'PPO': 'lightgrey', 'ActorCritic': 'lightcyan', 'GNN': 'lightyellow',
                    'TransferLearning': 'lightsalmon', 'RandomForest': 'lightpurple'
                }
                for model, preds in self.predictions[pair].items():
                    if self.plot_options[TRADING_PAIRS[0]][model].active and preds:
                        ax.plot(preds, label=model, color=colors.get(model, 'grey'), 
                                linestyle='--' if model == 'Ensemble' else '-', alpha=0.5 if model != 'Ensemble' else 1)
                ax2.plot(self.risk_levels[pair], label='Risk %', color='black', linestyle=':')
                ax2.plot(self.sentiment_scores[pair], label='Sentiment %', color='green', linestyle='-.')
                ax2.axhline(y=MAX_PORTFOLIO_RISK * 100, color='red', linestyle='--', label='Risk Limit')
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')
                ax.set_title(f"{pair} Price & Predictions")
                ax.set_ylabel("Price (USD)")
                ax2.set_ylabel("Portfolio Risk/Sentiment (%)")
                self.canvases[pair].draw()

    def update_ui_async_wrapper(self, dt):
        """Wrapper to schedule async UI updates."""
        asyncio.create_task(self.update_ui_async())

if __name__ == "__main__":
    from backend.api_handler import KrakenAPI
    from backend.ml_engine.ensemble import EnsembleModel
    from backend.risk_management import RiskManager
    from backend.strategies.manager import StrategyManager
    api = KrakenAPI(simulate=True)
    model = EnsembleModel(api=api)
    risk_manager = RiskManager(api)
    strategy_manager = StrategyManager(api, model, risk_manager)
    TradingApp(api, model, risk_manager, strategy_manager).run()