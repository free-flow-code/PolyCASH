## Polymarket traiding bot

### Инструкция по запуску
Установите зависимости:

```bash
pip install -r requirements.txt
```
Скопируйте `.env.example` в `.env` и заполните:
```bash
cp .env.example .env
```
Отредактируйте `.env`, добавив свои ключи.
Запустите бота в режиме бумажной торговли:
```bash
python main.py
```
Для реальной торговли: Установите `PAPER_TRADING=false` в `.env` и убедитесь, что у вас есть средства на адресе funder.