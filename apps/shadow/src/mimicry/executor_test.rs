use ghost_eyes::{Delivery, GhostError};

use super::retry_is_safe;
use crate::agent::tools::ax::fail;

#[test]
fn only_an_undelivered_failure_is_retried() {
    let undelivered = GhostError::not_supported("no action").with_delivery(Delivery::NotDelivered);
    assert!(retry_is_safe(&fail("ax_click", &undelivered).result));
    for delivery in [
        Delivery::DeliveryUncertain,
        Delivery::DeliveredUnverified,
        Delivery::Unknown,
    ] {
        let err = GhostError::internal("pressed, then failed").with_delivery(delivery);
        assert!(
            !retry_is_safe(&fail("ax_click", &err).result),
            "{delivery:?}"
        );
    }
    assert!(!retry_is_safe(&serde_json::json!({ "success": false })));
}
