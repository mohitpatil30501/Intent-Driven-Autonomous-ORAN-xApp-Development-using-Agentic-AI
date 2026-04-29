Module 3: Data Synthesizer (The Data Engineer)
What it does: This is a brilliant split. It generates two things based on the Cycle_Type:
Streaming Mock Data (Always generated): A CSV that mimics the real-time callbacks from the RIC (e.g., 100 rows of dl_aggr_tbs). Module 5 needs this to test its code.
Historical Training Data (Optional, only if ML): A larger dataset (e.g., CSV/JSON) representing past network behavior, used by Module 4 to train the model.