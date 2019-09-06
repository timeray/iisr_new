import datetime as dt

from glob import iglob
from pathlib import Path
from typing import Callable, List

from iisr.representation import Channel
from iisr.units import Frequency, TimeUnit

SEPARATOR = ','
DATE_RANGE_SEPARATOR = '/'
DATE_FMT = '%Y-%m-%d'


def option_parser_decorator(parser: Callable) -> Callable:
    def _parser_wrapper(option):
        if not option:
            raise ValueError('Empty option string')

        if option.lower() == 'none':
            return None
        else:
            return parser(option)

    return _parser_wrapper


@option_parser_decorator
def parse_optional_int(integer_string: str) -> int:
    return int(integer_string)


def parse_boolean(string: str) -> bool:
    string = string.lower()
    if string == 'true':
        return True
    elif string == 'false':
        return False
    else:
        raise ValueError('Incorrect boolean string: "{}"'.format(string))


@option_parser_decorator
def parse_list(input_list: str) -> List[str]:
    return [element.strip() for element in input_list.split(SEPARATOR)]


@option_parser_decorator
def parse_dates_ranges(dates_ranges_list: str) -> List[dt.date]:
    def _parse_date(date_str: str) -> dt.date:
        return dt.datetime.strptime(date_str, DATE_FMT).date()

    parsed_dates = []
    for date_range_str in parse_list(dates_ranges_list):
        if DATE_RANGE_SEPARATOR in date_range_str:
            start_date, stop_date = date_range_str.split('/')
            start_date = _parse_date(start_date)
            stop_date = _parse_date(stop_date)
            if stop_date <= start_date:
                raise ValueError('Stop date of the date range should be after start date, '
                                 '(expect start_date/stop_date)')
            for day in range((stop_date - start_date).days):
                parsed_dates.append(start_date + dt.timedelta(days=day))
        else:
            parsed_dates.append(_parse_date(date_range_str))
    return parsed_dates


@option_parser_decorator
def parse_path(paths: str) -> List[Path]:
    parsed_paths = []
    for path in parse_list(paths):
        if path != '~':
            parsed_paths.extend(Path(p) for p in iglob(path))
        else:
            parsed_paths.append(Path.home())
    return sorted(parsed_paths)


@option_parser_decorator
def parse_channels(channels: str) -> List[Channel]:
    return [Channel(int(ch)) for ch in parse_list(channels)]


@option_parser_decorator
def parse_frequency(frequencies: str) -> List[Frequency]:
    return [Frequency(float(freq), 'MHz') for freq in parse_list(frequencies)]


@option_parser_decorator
def parse_time_units(time_units_values_us: str) -> List[TimeUnit]:
    return [TimeUnit(float(val), 'us') for val in parse_list(time_units_values_us)]
