from matchcup.checkpointing import recipe_fingerprint
from matchcup.cross_encoder import CrossEncoderConfig, applied_ranking_weight


def test_default_reproduces_the_historical_hardcode():
    cfg = CrossEncoderConfig(model_name="x", ranking_weight=0.15)

    assert cfg.ranking_from_epoch == 1
    assert [applied_ranking_weight(cfg, epoch) for epoch in range(3)] == [0.0, 0.15, 0.15]


def test_a_threshold_above_the_epoch_count_keeps_the_loss_pure_bce():
    cfg = CrossEncoderConfig(
        model_name="x", ranking_weight=0.15, epochs=2, ranking_from_epoch=99
    )

    assert [applied_ranking_weight(cfg, epoch) for epoch in range(cfg.epochs)] == [0.0, 0.0]


def test_the_switch_on_epoch_is_part_of_resume_identity():
    a = recipe_fingerprint({"ranking_from_epoch": 1, "epochs": 2})
    b = recipe_fingerprint({"ranking_from_epoch": 99, "epochs": 2})

    assert a != b


def test_the_prevalence_sampler_is_still_selected_when_the_term_is_inert():
    """ranking_weight must stay 0.15: it is what selects the balanced sampler."""
    cfg = CrossEncoderConfig(model_name="x", ranking_weight=0.15, ranking_from_epoch=99)

    assert cfg.ranking_weight > 0
