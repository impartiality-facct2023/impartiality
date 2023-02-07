import jax 
from jax import numpy as jnp
import numpy as np

def finite_diff(func, x, eps=1e-4):
    return (func(x + eps/2) - func(x - eps/2))/eps

def query_fairPATE(sigma_threshold, sigma_gnmax, threshold, max_fairness_violation, min_group_count, subkey1, subkey2, raw_votes=None, targets=None, sensitives=None):
    # sigma_threshold = 50
    # sigma_gnmax = 5.0
    # threshold = 2.0
    # max_fairness_violation = 0.2
    # min_group_count = 50
    
    func = lambda x: 1.0
    num_classes = 10
    num_sensitive_attributes = 3
    num_samples = raw_votes.shape[0]
    votes=raw_votes
    noise_threshold = sigma_threshold * jax.random.normal(subkey1, [num_samples])
    noise_gnmax = sigma_gnmax * jax.random.normal(subkey2, [num_samples, num_classes])
    _shape = (1000, 10, 1)

    data = jax.lax.concatenate([jnp.broadcast_to(targets[:, None, None], _shape), 
                                  jnp.broadcast_to(sensitives[:, None, None], _shape), 
                                  jnp.broadcast_to(votes[:, :, None], _shape), 
                                  jnp.broadcast_to(noise_threshold[:, None, None], _shape),
                                  jnp.broadcast_to(noise_gnmax[:, :, None], _shape)], 2)

    def _calculate_gaps(sensitive_group_count, pos_classified_group_count):
        all_members = jnp.sum(sensitive_group_count)
        all_pos_classified_group_count = jnp.sum(pos_classified_group_count)
        dem_parity = jnp.divide(pos_classified_group_count, sensitive_group_count)
        others_count = all_members - sensitive_group_count
        others_pos_classified_group_count = all_pos_classified_group_count - pos_classified_group_count
        dem_parity_others = jnp.divide(others_pos_classified_group_count, others_count)
        gaps = dem_parity - dem_parity_others
        return gaps

    def _apply_fairness_constraint(pred, sensitive, answered, sensitive_group_count, pos_classified_group_count):
        gaps = _calculate_gaps(sensitive_group_count, pos_classified_group_count)
        sensitive_one_hot = (jnp.arange(num_sensitive_attributes) == sensitive).astype(float)
        answered = jax.lax.cond(sensitive_one_hot.dot(sensitive_group_count) < min_group_count, 
                             (answered, pred, gaps), lambda x: x[0],
                             (answered, pred, gaps), lambda x: jax.lax.cond(x[1] == 0.0, 
                                                                   x, lambda y: y[0],
                                                                   x, lambda y: jax.lax.cond(sensitive_one_hot.dot(y[2]) < max_fairness_violation,
                                                                                                     y, lambda z: z[0],
                                                                                                     y, lambda z: 0.0)
                                                                  )
                           )

        sensitive_group_count = jax.lax.cond(answered == 1.,
                                         sensitive_group_count, lambda x: x+sensitive_one_hot,
                                         sensitive_group_count, lambda x: x)

        pos_classified_group_count = jax.lax.cond(answered == 1.,
                                         (pos_classified_group_count, pred), lambda x: x[0] + sensitive_one_hot * jax.lax.cond(x[1]==1., 1., lambda x: x, 0., lambda x:x), 
                                         (pos_classified_group_count, pred), lambda x: x[0])

        return answered, sensitive_group_count, pos_classified_group_count

    def _predict(output, _data):
        acc, sensitive_group_count, pos_classified_group_count = output
        _target = _data[0, 0]
        _sensitive = _data[0, 1]
        _vote = _data[:, 2]
        _noise_threshold = _data[0, 3]
        _noise_gnmax = _data[:, 4]
        
        vote_count = _vote.max()
        noisy_vote_count = vote_count + _noise_threshold
        answered = jax.lax.cond(noisy_vote_count > threshold, threshold, func, threshold, lambda x: 0.0)
        pred = (_vote + _noise_gnmax).argmax()
        answered, sensitive_group_count, pos_classified_group_count = \
                            _apply_fairness_constraint(pred, _sensitive, answered, sensitive_group_count, pos_classified_group_count)
        acc = acc + answered * (pred==_target).astype(int)
        output = acc, sensitive_group_count, pos_classified_group_count
        return output, answered

    output, answered = jax.lax.scan(_predict, (jnp.zeros((1,)), jnp.zeros((num_sensitive_attributes,)), jnp.zeros((num_sensitive_attributes,))), data, length=len(votes))
    accuracy = output[0][0]/num_samples
    gaps = _calculate_gaps(*output[1:])
    return accuracy, answered, gaps


if __name__ == "__main__":
    path = "./20-models/"
    targets = np.load(path + "targets-multiclass-model(1)-labels-(mode:random)-(threshold:300.0)-(sigma-gnmax:40.0)-(sigma-threshold:200.0)-(budget:16.00)-transfer-.npy").astype(float)
    raw_votes = np.load(path + "model(1)-raw-votes-mode-random-vote-type-discrete.npy").astype(float)
    sensitives = np.random.choice(np.arange(3).astype(float), (1000,), p=[0.1, 0.3, 0.6])
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    results = query_fairPATE(50.0, 5.0, 0.22, 0.2, 50, subkey1, subkey2, raw_votes=raw_votes, targets=targets, sensitives=sensitives)
    print(results)