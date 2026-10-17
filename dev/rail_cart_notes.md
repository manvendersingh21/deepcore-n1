# Actuated mine rail carts

`ui.rail_carts.add_rail_carts(xml)` adds three open rail carts on a shared
visual track. Each cart starts parked at travel position zero; with actuator
control at zero, joint damping holds it there. The native controller can drive
`cart_i_drive` negatively after loading two chunks; its slide joint
`cart_i_travel` is limited to [-3, 0] m along X. Cart 0 is at the
front of the row and its departure travels away from the other carts. Only each
cart's floor and four walls collide, so physical debris can settle into the
tubs. Wheels, axles, rails, sleepers, buffers, and safety markings are
noncolliding presentation geometry.

The cart floors are named `cart_0_floor`, `cart_1_floor`, and `cart_2_floor`;
these are stable geom anchors for rendering labels and cart occupancy.
`CART_CENTERS`, `CART_CAPACITY`, `cart_interior_bounds(index)`,
`cart_slot_positions(index)`, and `cart_layout_metadata()` expose the layout to
the fleet controller and UI. Each cart has two slots at y offsets ±0.085 m;
the default chunk center is 0.449 m high, placing the 23 mm half-height debris
on the 0.426 m physical floor. The interior is 0.36 × 0.36 m with a 0.56 m rim.
The row's rails extend approximately from x=-4 m to x=2.6 m to cover all three
travel paths. Pass the measured joint position as `travel_x` to the bounds and
slot helpers after a cart moves.

This models individual actuated cart departures after loading; it does not
model coupled train operation.
