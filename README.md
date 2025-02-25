### Инструкция к запуску проекта
1. Запустить ноутбук notebooks/creating datasets.ipynb, он создает датасет, делит его на трейн/тест (если исходно не поделен), тестом Колмогорова-Смирнова проверяет различие в распределениях фичей, с помощью модели пытается отличить трейн и тест.
2. notebooks/example of working.ipynb - пример запуска теста модели
3. Для добавления новой icausaml модели надо в src.models.ICAusalML.Models добавить модель, унаследованную от ICAusalML. Также в src.factory добавить соответствующую фабрику, в utils.generate_configs добавить генератор конфига 

# План текущий момент
1. T model, S model знаю понятно
   https://habr.com/ru/companies/ru_mts/articles/485980/
   Список статей с хабра или отсюда https://causalml.readthedocs.io/en/latest/methodology.html
2. R model, X model изучить
   https://habr.com/ru/companies/glowbyte/articles/686398/

3. Из EconML оставшиеся + аплифт леса
https://habr.com/ru/companies/ru_mts/articles/485976/

4. DragonNet, DESCN

5. Для рл:
Посмотреть курс шадовский
https://github.com/oppsitre/RLift/blob/master/src/PolicyGradient.py
DQN https://www.hse.ru/en/en/ma/sltheory/students/diplomas/page3.html/641461071

6. Со звездочкой графовые нейронки https://arxiv.org/pdf/2311.08434


# UpliftModelingResearch

Работа будет посвящена поиску оптимальной модели по трейдофу между качеством и временем инференса.
Качество: AUUC (area under uplift curve), Uplift@k.
 
План примерно такой:
1. Датасеты

Критео (мало фичей много данных) https://huggingface.co/datasets/criteo/criteo-uplift https://www.uplift-modeling.com/en/latest/api/datasets/fetch_criteo.html

датасет от x5 https://ods.ai/competitions/x5-retailhero-uplift-modeling/data https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html#x5-retailhero-uplift-modeling-dataset

лента (50 фичей) https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html#lenta-uplift-modeling-dataset

LAZADA: https://arxiv.org/pdf/2207.09920
Бенчмарки есть на критео и ласаде 
https://arxiv.org/pdf/2406.00335 (критео и ласада)
https://arxiv.org/pdf/2207.09920 (ласада и странный искуственный датасет) https://github.com/kailiang-zhong/DESCN/tree/main/data/Lazada_dataset

2. Попробовать:

SMODELING, TMODELING, X-LEARNER, R-learner на бустингах/полносвязанных нейронках.

3. Разобраться с аплифт деревьями и попробовать случайный лес на аплифт деревьях (Uplift random forest от causalML) (вроде это SOTA)

4. Разобраться с нейронками, по которым я нашел статьи (мб все не получится). 

TARNET, DRAGONET, Графовые нейронки, Rl.
https://arxiv.org/pdf/2311.14994
https://arxiv.org/pdf/2011.00041
https://arxiv.org/abs/2311.08434
https://www.hse.ru/en/en/ma/sltheory/students/diplomas/page3.html/641461071 (нашел в вышке работу с rl в аплифте, мб свяжусь с автором)
https://towardsdatascience.com/tarnet-and-dragonnet-causal-inference-between-s-and-t-learners-0444b8cc65bd 

6. Сравнить модели по качеству (AUUC, Uplift@k, loss) 
7. Попробовать ускорить топовые модели.

Запрунить деревья в аплифт-лесе, скомпрессировать нейронки (knoiwledge distilaltion, pruning, quantization)

8. Сравнить модели по качеству (AUUC, Uplift@k, loss) и по времени инференса

Аплифт выглядит очень актуальной темой + кажется маловато ресерча на эту тему
