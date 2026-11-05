"""Aprendizaje por imitación (Behavior Cloning / DAgger) y fine-tuning SAC para SolverBot.

La política aprendida solo ve la observación de Gymnasium (31 valores, object_source: perception). El experto
guionado usa información privilegiada SOLO para generar etiquetas; esa información se guarda con prefijo `priv_` en
las demostraciones y nunca entra al modelo (ver dataset.POLICY_INPUT_KEY y las pruebas test_imitation.py).
"""
