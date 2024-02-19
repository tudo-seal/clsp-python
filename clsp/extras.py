"""
Some extra functionality, that may or may not be of use.

These are considered (mostly) untested and highly unstable. They can disappear without notice from
one day to the other. Do not rely on them.

Feedback on their usefulness is appreciated though.
"""

from collections.abc import Callable
from typing import Any, Optional


def configurable(
    combinator: Optional[Callable[..., Any]] = None,
    /,
    *,
    forward_config: bool = True,
    use_config: bool = True,
) -> Callable[..., Any]:
    """Makes a combinator configurable.

    A configurable combinator takes an extra configuration argument after evaluation,
    and forwards this argument to all its children. This extra argument is always the
    last argument of the combinator.

    Example:
      Given the configurable combinators

      ```
      @configurable
      def modulo_add(a, b, config):
        return (a + b) % config

      @configurable
      def four(config):
        return 4 % config
      ```

      With annotated types A -> A and A respectively (where A is some constructor).

      An interpreted term of type A is now a function, that takes a config (in this example simply
      an number) and applies the modulo operation at each of its levels.

      In particular a term `modulo_add(four, modulo_add(four, four))` still needs this value to
      yield a result. When given a number (e.g. 3), this would evaluate to:

      ```
      modulo_add(four, modulo_add(four, four))(3)
      = (four(3) + modulo_add(four, four)(3)) % 3
      = (4 % 3 + (four(3) + four(3) % 3)) % 3
      = (1 + (4 % 3 + 4 + 3)) % 3
      = (1 + 1 + 1) % 3
      = 0
      ```

    Note:
      The implementation of the combinator itself does not need to forward the configuration
      argument manually. This is completely handled by the decorator.

    Args:
        combinator: The combinator function to make configurable. If None, returns
            the decorator itself.
        forward_config: Whether to forward the configuration argument to children
            functions.
        use_config: Whether to use the configuration argument within the combinator
            itself.

    Returns:
        The decorated combinator, or the decorator itself if no combinator is
        provided.
    """

    def decorate(combinator: Callable[..., Any]) -> Callable[..., Any]:
        def inner(*args: Any) -> Callable[[Any], Any]:
            def config_forwarder(config: Any) -> Any:
                configured_args = (
                    arg(config) if callable(arg) and forward_config else arg for arg in args
                )

                if use_config:
                    return combinator(*configured_args, config)
                else:
                    return combinator(*configured_args)

            return config_forwarder

        return inner

    if combinator is None:
        return decorate
    else:
        return decorate(combinator)


if __name__ == "__main__":

    @configurable
    def test(a: int, b: int, c: int) -> int:
        return a + b + c

    print(test(1, 2)(3))
    print(test(test(1, 2), 3)(4))
