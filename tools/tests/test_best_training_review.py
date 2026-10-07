"""Manual review is recoverable, scoped and cannot bypass training gates."""
import json
from pathlib import Path
import threading
import uuid

import pytest

from test_best_training_faceset import backend, files_sha, make_image, ready_plan, fake_analysis
from core.best_training_faceset import manual_review_summary, review_eligibility


def operation():
    return str(uuid.uuid4())


def review_ready(tmp_path, monkeypatch, *, published=False):
    source, parent, initial = ready_plan(tmp_path, monkeypatch)
    publication = backend.publish_plan(parent, tmp_path / 'output') if published else None
    before = files_sha(Path(parent))
    child = backend.create_review_plan(parent, expected_revision=0, request_id=operation())
    return source, parent, initial, child['planDirectory'], child, before, publication


def decide(plan, revision, choice, members, *, refs=('0.jpg',), request_id=None):
    return backend.apply_review_decision(plan, expected_revision=revision, request_id=request_id or operation(),
        expected_references=refs, decision=choice, members=members)


def test_new_review_keeps_original_published_plan_and_receipt_immutable(tmp_path, monkeypatch):
    source, parent, original, plan, child, before, published = review_ready(tmp_path, monkeypatch, published=True)
    receipt_before = Path(published['receiptPath']).read_bytes()
    assert child['parentPlanId'] == Path(parent).name and child['reviewRootPlanId'] == Path(parent).name
    assert child['reviewRevision'] == 0 and child['reviewVersion'] == 1
    assert child['datasetFingerprint'] == original['datasetFingerprint'] and child['sidecarFingerprint'] == original['sidecarFingerprint']
    assert child['reviewSummary']['adequateReserveCount'] == 1 and child['reviewSummary']['pendingCount'] == 0
    old = next(item for item in child['items'] if item['member'] == '1.jpg')
    result = decide(plan, 0, 'keep', ['1.jpg'])
    kept = next(item for item in result['items'] if item['member'] == '1.jpg')
    assert kept['status'] == 'selected' and kept['manualDecision']['decision'] == 'keep'
    assert kept['algorithmSuggestion'] == old['algorithmSuggestion'] and kept['algorithmSuggestion']['classification'] == 'adequate-reserve'
    assert result['selection']['counts'] == {'selected': 3, 'review': 0, 'rejected': 0}
    assert result['selection']['algorithmCounts'] == original['selection']['counts']
    assert files_sha(Path(parent)) == before and Path(published['receiptPath']).read_bytes() == receipt_before
    assert backend.verify_source(plan)['sourceValid']


def test_batch_decisions_undo_restart_idempotency_and_stale_revision(tmp_path, monkeypatch):
    source, parent, _, plan, _, before, _ = review_ready(tmp_path, monkeypatch)
    request = operation()
    first = decide(plan, 0, 'exclude', ['0.jpg', 'rare.jpg'], request_id=request)
    assert first['reviewRevision'] == 1 and first['reviewSummary']['decidedCount'] == 2
    resumed = backend.inspect_plan(Path(plan))
    assert resumed['reviewSummary']['canUndo'] and resumed['selection']['counts']['rejected'] == 2
    assert decide(plan, 0, 'exclude', ['rare.jpg', '0.jpg'], request_id=request)['reviewOperation']['reused']
    with pytest.raises(backend.PlanConflict, match='UUID'): decide(plan, 0, 'keep', ['0.jpg'], request_id=request)
    with pytest.raises(backend.PlanConflict, match='版本已变化'): decide(plan, 0, 'defer', ['1.jpg'])
    second = decide(plan, 1, 'defer', ['1.jpg'])
    assert second['reviewSummary']['deferredCount'] == 1 and second['reviewSummary']['adequateReserveCount'] == 0
    undone = backend.undo_review_decision(plan, expected_revision=2, request_id=operation(), expected_references=['0.jpg'])
    assert undone['reviewRevision'] == 3 and undone['reviewSummary']['deferredCount'] == 0
    undone = backend.undo_review_decision(plan, expected_revision=3, request_id=operation(), expected_references=['0.jpg'])
    assert undone['reviewRevision'] == 4 and undone['reviewSummary']['decidedCount'] == 0 and not undone['reviewSummary']['canUndo']
    assert undone['selection']['counts'] == {'selected': 2, 'review': 1, 'rejected': 0}
    assert files_sha(Path(parent)) == before


def test_pending_excludes_reserves_and_keeps_deferred_real_review_pending():
    records = [{'status': 'review', 'classification': 'needs-review'}, {'status': 'review', 'classification': 'adequate-reserve'},
        {'status': 'review', 'classification': 'manual-defer', 'algorithmSuggestion': {'classification': 'needs-review'}, 'manualDecision': {'decision': 'defer'}},
        {'status': 'rejected', 'classification': 'manual-exclude', 'algorithmSuggestion': {'classification': 'needs-review'}, 'manualDecision': {'decision': 'exclude'}}]
    result = manual_review_summary(records)
    assert result['pendingCount'] == 2 and result['adequateReserveCount'] == 1
    assert result['deferredCount'] == 1 and result['decidedCount'] == 1


@pytest.mark.parametrize('failure', ['hard', 'identity', 'quality-floor', 'missing-quality', 'missing-coverage', 'references-conflict', 'references-empty'])
def test_keep_rejects_hard_and_identity_quality_evidence_gaps(tmp_path, monkeypatch, failure):
    _, _, _, plan, _, _, _ = review_ready(tmp_path, monkeypatch)
    with backend._connect(Path(plan)) as connection:
        row = connection.execute("SELECT decision FROM records WHERE member='1.jpg'").fetchone()
        record = json.loads(row[0])
        if failure == 'hard': record['hardReasons'] = ['DFL元数据损坏']
        elif failure == 'identity': record['identity']['maximumSimilarity'] = .1
        elif failure == 'quality-floor': record['quality']['score'] = 0.
        elif failure == 'missing-quality': record.pop('quality')
        elif failure == 'missing-coverage': record.pop('coverage')
        connection.execute("UPDATE records SET decision=? WHERE member='1.jpg'", (json.dumps(record),))
    if failure in ('references-conflict', 'references-empty'):
        path = Path(plan) / 'plan.json'; metadata = json.loads(path.read_text(encoding='utf-8'))
        if failure == 'references-empty': metadata['confirmedReferenceMembers'] = []
        else:
            metadata['selection']['identityReferencesConflict'] = True
            with backend._connect(Path(plan)) as connection:
                patch = json.loads(connection.execute('SELECT metadata FROM review_state').fetchone()[0])
                patch['selection']['identityReferencesConflict'] = True
                connection.execute('UPDATE review_state SET metadata=?', (json.dumps(patch),))
        backend._json(path, metadata)
    before = backend.inspect_plan(plan)
    assert not next(item for item in before['items'] if item['member'] == '1.jpg')['reviewEligibility']['canKeep']
    with pytest.raises(backend.PlanConflict, match='不能人工保留'):
        decide(plan, 0, 'keep', ['0.jpg', '1.jpg'], refs=() if failure == 'references-empty' else ('0.jpg',))
    after = backend.inspect_plan(plan)
    assert after['reviewRevision'] == 0 and after['selection'] == before['selection']
    assert all(item['manualDecision'] is None for item in after['items'])


def test_real_missing_dfl_cannot_be_kept_even_if_stored_suggestion_was_corrupt(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    make_image(source / '0.jpg'); make_image(source / 'plain.jpg', metadata=False)
    monkeypatch.setattr(backend, 'analyze_image', fake_analysis)
    parent = backend.create_plan(source, tmp_path / 'plans', target_count=10, confirmed_reference_members=['0.jpg'])['planDirectory']
    backend.analyze_batch(parent, models=object()); backend.finalize_plan(parent)
    plan = backend.create_review_plan(parent, expected_revision=0, request_id=operation())['planDirectory']
    with pytest.raises(backend.PlanConflict, match='DFL元数据'): decide(plan, 0, 'keep', ['plain.jpg'])
    assert backend.inspect_plan(plan)['reviewRevision'] == 0


def test_reference_change_source_change_and_sidecar_change_are_rejected(tmp_path, monkeypatch):
    source, _, _, plan, _, _, _ = review_ready(tmp_path, monkeypatch)
    with pytest.raises(backend.PlanConflict, match='身份参照已变化'): decide(plan, 0, 'exclude', ['1.jpg'], refs=['rare.jpg'])
    with pytest.raises(backend.PlanConflict, match='重新定义身份'): backend.confirm_references(plan, ['rare.jpg'])
    with pytest.raises(backend.PlanConflict, match='保留原算法'): backend.finalize_plan(plan)
    sidecar = source / '0.jpg.landmarks.json'; sidecar.write_text('{"changed":true}')
    before = files_sha(Path(plan))
    with pytest.raises(backend.PlanConflict, match='sidecar已变化'): decide(plan, 0, 'exclude', ['1.jpg'])
    with pytest.raises(backend.PlanConflict, match='sidecar已变化'): backend.verify_source(plan)
    assert files_sha(Path(plan)) == before


def test_quantity_upper_limit_and_batch_bounds_remain_enforced(tmp_path, monkeypatch):
    _, _, _, plan, _, _, _ = review_ready(tmp_path, monkeypatch)
    path = Path(plan) / 'plan.json'; metadata = json.loads(path.read_text(encoding='utf-8')); metadata['targetCount'] = 2; backend._json(path, metadata)
    with pytest.raises(backend.PlanConflict, match='数量上限'): decide(plan, 0, 'keep', ['1.jpg'])
    assert backend.inspect_plan(plan)['reviewRevision'] == 0
    with pytest.raises(ValueError, match='1..500'): decide(plan, 0, 'exclude', ['0.jpg'] * 501)
    with pytest.raises(ValueError, match='1..500'): decide(plan, 0, 'exclude', ['0.jpg', '0.jpg'])


def test_interrupted_json_mirror_write_rolls_back_and_recovers(tmp_path, monkeypatch):
    _, _, _, plan, _, _, _ = review_ready(tmp_path, monkeypatch)
    save = backend._json
    def interrupted(path, value):
        save(path, value)
        if path.name == 'plan.json' and value.get('reviewRevision') == 1: raise OSError('simulated interruption after mirror write')
    monkeypatch.setattr(backend, '_json', interrupted)
    with pytest.raises(OSError): decide(plan, 0, 'exclude', ['0.jpg', 'rare.jpg'])
    monkeypatch.setattr(backend, '_json', save)
    restored = backend.inspect_plan(plan)
    assert restored['reviewRevision'] == 0 and restored['reviewSummary']['decidedCount'] == 0
    assert json.loads((Path(plan) / 'plan.json').read_text(encoding='utf-8'))['reviewRevision'] == 0


def test_concurrent_edits_compare_and_swap_only_one_revision(tmp_path, monkeypatch):
    _, _, _, plan, _, _, _ = review_ready(tmp_path, monkeypatch)
    barrier = threading.Barrier(2); outcomes = []
    def submit(member):
        barrier.wait()
        try: outcomes.append(decide(plan, 0, 'exclude', [member]))
        except backend.PlanConflict as error: outcomes.append(error)
    threads = [threading.Thread(target=submit, args=(member,)) for member in ('0.jpg', 'rare.jpg')]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=20)
    assert len(outcomes) == 2 and sum(isinstance(result, dict) for result in outcomes) == 1
    assert sum(isinstance(result, backend.PlanConflict) for result in outcomes) == 1
    result = backend.inspect_plan(plan)
    assert result['reviewRevision'] == 1 and result['reviewSummary']['decidedCount'] == 1


def test_reader_waiting_for_publication_lock_never_restores_prepublication_state(tmp_path, monkeypatch):
    _, _, _, plan, _, _, _ = review_ready(tmp_path, monkeypatch)
    original_connect = backend._connect; waiting = threading.Event(); results = []
    def signal_connect(directory):
        connection = original_connect(directory)
        if threading.current_thread().name == 'review-reader': waiting.set()
        return connection
    monkeypatch.setattr(backend, '_connect', signal_connect)
    writer = original_connect(Path(plan)); writer.execute('BEGIN IMMEDIATE')
    reader = threading.Thread(name='review-reader', target=lambda: results.append(backend.inspect_plan(plan)))
    reader.start()
    assert waiting.wait(10)
    path = Path(plan) / 'plan.json'; metadata = json.loads(path.read_text(encoding='utf-8'))
    metadata['state'] = 'published'; backend._json(path, metadata)
    writer.commit(); writer.close(); reader.join(timeout=15)
    assert len(results) == 1 and results[0]['state'] == 'published'
    assert json.loads(path.read_text(encoding='utf-8'))['state'] == 'published'


def test_review_publish_requires_current_revision_freezes_then_forks_again(tmp_path, monkeypatch):
    source, _, _, plan, _, _, _ = review_ready(tmp_path, monkeypatch)
    before = files_sha(source)
    decide(plan, 0, 'keep', ['1.jpg'])
    with pytest.raises(backend.PlanConflict, match='须确认当前版本'): backend.publish_plan(plan, tmp_path / 'output', dry_run=True)
    with pytest.raises(backend.PlanConflict, match='版本已变化'): backend.publish_plan(plan, tmp_path / 'output', dry_run=True, expected_revision=0)
    dry = backend.publish_plan(plan, tmp_path / 'output', dry_run=True, expected_revision=1)
    publication = backend.publish_plan(plan, tmp_path / 'output', expected_revision=1)
    assert dry['counts']['selected'] == 3
    receipt = json.loads(Path(publication['receiptPath']).read_text(encoding='utf-8'))
    assert receipt['reviewRevision'] == 1 and receipt['parentPlanId']
    assert next(entry for entry in receipt['entries'] if entry['member'] == '1.jpg')['manualDecision']['decision'] == 'keep'
    with pytest.raises(backend.PlanConflict, match='不能改写'): decide(plan, 1, 'exclude', ['1.jpg'])
    new = backend.create_review_plan(plan, expected_revision=1, request_id=operation())
    assert new['reviewVersion'] == 2 and new['reviewRevision'] == 0
    assert backend.recover_publication(publication['receiptPath'], dry_run=True)['conflicts'] == []
    assert backend.recover_publication(publication['receiptPath'])['state'] == 'withdrawn'
    assert files_sha(source) == before


def test_review_create_retry_same_uuid_and_old_expected_revision_are_safe(tmp_path, monkeypatch):
    _, parent, _, _, _, _, _ = review_ready(tmp_path, monkeypatch)
    request = operation()
    first = backend.create_review_plan(parent, expected_revision=0, request_id=request)
    second = backend.create_review_plan(parent, expected_revision=0, request_id=request)
    assert second['planId'] == first['planId'] and second['reviewOperation']['reused']
    with pytest.raises(backend.PlanConflict, match='UUID'): backend.create_review_plan(parent, expected_revision=1, request_id=request)


def test_concurrent_duplicate_create_adopts_one_child_without_repeating_the_fork(tmp_path, monkeypatch):
    _, parent, _ = ready_plan(tmp_path, monkeypatch)
    barrier = threading.Barrier(2); results = []; request = operation()
    def create():
        barrier.wait()
        try: results.append(backend.create_review_plan(parent, expected_revision=0, request_id=request))
        except BaseException as error: results.append(error)
    threads = [threading.Thread(target=create) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=20)
    assert len(results) == 2 and all(isinstance(result, dict) for result in results)
    assert results[0]['planId'] == results[1]['planId']
    assert sum(result['reviewOperation']['reused'] for result in results) == 1
    assert len(list(Path(parent).parent.glob('plan-*'))) == 2
