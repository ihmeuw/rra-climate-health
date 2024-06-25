from typing import ParamSpec, TypeVar, Callable, Any, List

import click
from rra_tools.cli_tools import (
    RUN_ALL,
    ClickOption,
    with_choice,
    with_debugger,
    with_input_directory,
    with_num_cores,
    with_output_directory,
    with_progress_bar,
    with_queue,
    with_verbose,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")


VALID_MEASURES = ["wasting", "stunting"]


def get_choice_callback(
    allow_all: bool,
    choices: list[str],
) -> Callable[[Any, Any, Any], list[str] | str]:
    if allow_all:
        return lambda ctx, param, value: choices if value == RUN_ALL else [value]
    else:
        return lambda ctx, param, value: value


def with_measure(
    *,
    choices: list[str] = VALID_MEASURES,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "measure",
        "m",
        allow_all=allow_all,
        choices=choices,
        help="The nutrition measure to run.",
        callback=get_choice_callback(allow_all, choices),
    )


def with_model_id() -> ClickOption[_P, _T]:
    return click.option(
        "--model-id",
        help="A string that identifies the particular model to run.",
        type=str,
    )


VALID_FHS_LOCATION_IDS = [
    '7', '8', '10', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22', '23', '24', '25', '26', '27', '28', '29',
    '30', '33', '34', '35', '36', '37', '38', '39', '40', '41', '43', '44', '45', '46', '47', '48', '49', '50', '51',
    '52', '53', '54', '55', '57', '58', '59', '60', '61', '62', '63', '66', '67', '68', '69', '71', '72', '74', '75',
    '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '91', '92', '93', '94', '97',
    '98', '99', '101', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117',
    '118', '119', '121', '122', '123', '125', '126', '127', '128', '129', '131', '132', '133', '136', '139', '140',
    '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156',
    '157', '160', '161', '162', '164', '168', '169', '170', '171', '172', '173', '175', '176', '177', '178', '180',
    '181', '182', '183', '184', '185', '186', '187', '189', '190', '191', '193', '194', '195', '196', '197', '198',
    '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '215', '216',
    '217', '218', '298', '305', '320', '349', '351', '354', '361', '367', '369', '374', '376', '380', '385', '393',
    '396', '413', '416', '422', '433', '434', '435', '491', '492', '493', '494', '495', '496', '497', '498', '499',
    '500', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512', '513', '514', '515',
    '516', '517', '518', '519', '520', '521', '522', '523', '524', '525', '526', '527', '528', '529', '530', '531',
    '532', '533', '534', '535', '536', '537', '538', '539', '540', '541', '542', '543', '544', '545', '546', '547',
    '548', '549', '550', '551', '552', '553', '554', '555', '556', '557', '558', '559', '560', '561', '562', '563',
    '564', '565', '566', '567', '568', '569', '570', '571', '572', '573', '4636', '4643', '4644', '4645', '4646',
    '4647', '4648', '4649', '4650', '4651', '4652', '4653', '4654', '4655', '4656', '4657', '4658', '4659', '4660',
    '4661', '4662', '4663', '4664', '4665', '4666', '4667', '4668', '4669', '4670', '4671', '4672', '4673', '4674',
    '4709', '4710', '4711', '4712', '4713', '4714', '4715', '4716', '4717', '4718', '4719', '4720', '4721', '4722',
    '4723', '4724', '4725', '4726', '4727', '4728', '4729', '4730', '4731', '4732', '4733', '4734', '4735', '4736',
    '4737', '4738', '4739', '4740', '4741', '4742', '4749', '4750', '4751', '4752', '4753', '4754', '4755', '4756',
    '4757', '4758', '4759', '4760', '4761', '4762', '4763', '4764', '4765', '4766', '4767', '4768', '4769', '4770',
    '4771', '4772', '4773', '4774', '4775', '4776', '4841', '4842', '4843', '4844', '4846', '4849', '4850', '4851',
    '4852', '4853', '4854', '4855', '4856', '4857', '4859', '4860', '4861', '4862', '4863', '4864', '4865', '4867',
    '4868', '4869', '4870', '4871', '4872', '4873', '4874', '4875', '4910', '4920', '4923', '4926', '25318',
    '25319', '25320', '25321', '25322', '25323', '25324', '25325', '25326', '25327', '25328', '25329', '25330',
    '25331', '25332', '25333', '25334', '25335', '25336', '25337', '25338', '25339', '25340', '25341', '25342',
    '25343', '25344', '25345', '25346', '25347', '25348', '25349', '25350', '25351', '25352', '25353', '25354',
    '44538', '44852', '44853', '44854', '44855', '44856', '44857', '44858', '44859', '44860', '44861', '44862',
    '53432', '53615', '53616', '53617', '53618', '53619', '53620', '53621', '60132', '60133', '60134', '60135',
    '60136', '60137',
]
 

def with_location_id(
    *,
    choices: list[str] = VALID_FHS_LOCATION_IDS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "location-id",
        "l",
        allow_all=allow_all,
        choices=choices,
        help="The location ID to run.",
        callback=get_choice_callback(allow_all, choices),
    )


VALID_CMIP6_SCENARIOS = [
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
]


def with_cmip6_scenario(
    *,
    choices: list[str] = VALID_CMIP6_SCENARIOS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "cmip6-scenario",
        "c",
        allow_all=allow_all,
        choices=choices,
        help="The CMIP6 scenario to run.",
        callback=get_choice_callback(allow_all, choices),
    )


VALID_SEX_IDS = ["1", "2"]


def with_sex_id(
    *,
    choices: list[str] = VALID_SEX_IDS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "sex-id",
        "s",
        allow_all=allow_all,
        choices=choices,
        help="The sex ID to run. 1 is Male and 2 is Female.",
        callback=get_choice_callback(allow_all, choices),
    )


VALID_AGE_GROUP_IDS = ["4", "5"]


def with_age_group_id(
    *,
    choices: list[str] = VALID_AGE_GROUP_IDS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "age-group-id",
        "a",
        allow_all=allow_all,
        choices=choices,
        help="The age group ID to run.",
        callback=get_choice_callback(allow_all, choices),
    )


VALID_PREDICTION_YEARS = [str(year) for year in range(2022, 2101)]


def with_year(
    *,
    choices: list[str] = VALID_PREDICTION_YEARS,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "year",
        "y",
        allow_all=allow_all,
        choices=choices,
        help="The year to run.",
        callback=get_choice_callback(allow_all, choices),
    )


def with_overwrite() -> ClickOption[_P, _T]:
    return click.option(
        "--overwrite",
        help="Overwrite existing files.",
        is_flag=True,
    )
