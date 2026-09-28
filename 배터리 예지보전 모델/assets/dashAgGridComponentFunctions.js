var dagcomponentfuncs = window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

dagcomponentfuncs.EventDetailButton = function (props) {
    function selectEvent() {
        props.setData({ action: "open", eventId: props.data.event_id });
    }
    return React.createElement(
        "button",
        { onClick: selectEvent, className: "event-detail-button", "aria-label": props.data.event_label + " 상세 보기" },
        "상세"
    );
};
