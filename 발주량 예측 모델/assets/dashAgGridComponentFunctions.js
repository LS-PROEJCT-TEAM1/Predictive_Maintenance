var dagcomponentfuncs = window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

dagcomponentfuncs.DetailButton = function (props) {
    function handleClick() {
        props.setData({
            part_number: props.data.part_number,
            target_date: props.data.target_date
        });
    }
    return React.createElement(
        'button',
        {
            className: 'detail-grid-button',
            onClick: handleClick,
            type: 'button',
            'aria-label': props.data.part_number + ' 상세 검토'
        },
        '상세 검토'
    );
};
